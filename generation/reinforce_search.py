"""
REINFORCE binary path search for FLUX.2-klein-base-9B.

Instead of exhaustively generating all 2^N_BITS paths, learns Bernoulli
parameters for each bit position via REINFORCE policy gradient.

Usage:
  python generation/reinforce_search.py \
      --source_prompt "a tabby cat walking on stone pavement, photo" \
      --target_prompt "a golden retriever dog walking on stone pavement, photo" \
      --seg_prompt "cat" \
      --output_dir /path/to/output \
      --gpu 0 --n_bits 14 --num_episodes 500 --batch_size 8
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.distributions import Bernoulli

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"


def mask_to_int_batch(masks):
    """Convert (B, n_bits) binary array to integer array (MSB-first)."""
    bits = np.arange(masks.shape[1] - 1, -1, -1)
    return (masks << bits).sum(axis=1)


# ────────────────────────────────────────────
# SSIM (from metrics/calc_seg_metrics.py)
# ────────────────────────────────────────────

def _gaussian_kernel_2d(size=11, sigma=1.5, channels=3, device="cpu", dtype=torch.float32):
    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k2d = g[:, None] * g[None, :]
    k2d = k2d / k2d.sum()
    return k2d.expand(channels, 1, size, size).contiguous()


def ssim_map(img1, img2, kernel, pad):
    """Per-pixel SSIM between (B,C,H,W) tensors in [0,1]."""
    C = img1.shape[1]
    mu1 = F.conv2d(img1, kernel, padding=pad, groups=C)
    mu2 = F.conv2d(img2, kernel, padding=pad, groups=C)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, kernel, padding=pad, groups=C) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, kernel, padding=pad, groups=C) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, kernel, padding=pad, groups=C) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
        ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return s.mean(dim=1, keepdim=True)


# ────────────────────────────────────────────
# Vision model GPU-side preprocessing
# ────────────────────────────────────────────

# Normalization stats per family.
# SigLIP / SigLIP2 use simple [-1, 1] range (mean/std = 0.5), CLIP uses ImageNet stats.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
SIGLIP_MEAN = [0.5, 0.5, 0.5]
SIGLIP_STD = [0.5, 0.5, 0.5]


def build_clip_transform(device, dtype=torch.float32):
    """Back-compat wrapper returning CLIP (ViT-B/32) normalization constants."""
    mean = torch.tensor(CLIP_MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def build_vision_transform(device, model_family="siglip2", dtype=torch.float32):
    """Return (mean, std) tensors for a given vision model family ('clip' or 'siglip2')."""
    if model_family == "clip":
        mean_vals, std_vals = CLIP_MEAN, CLIP_STD
    else:  # siglip / siglip2
        mean_vals, std_vals = SIGLIP_MEAN, SIGLIP_STD
    mean = torch.tensor(mean_vals, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(std_vals, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def clip_preprocess_gpu(images_01, clip_mean, clip_std, target_size=224):
    """Back-compat wrapper — hardcoded 224 for CLIP ViT-B/32."""
    x = F.interpolate(images_01, size=(target_size, target_size),
                      mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - clip_mean) / clip_std


def vision_preprocess_gpu(images_01, mean, std, target_size):
    """GPU-side resize + normalize for any vision model. Input: (B,3,H,W) in [0,1]."""
    x = F.interpolate(images_01, size=(target_size, target_size),
                      mode="bicubic", align_corners=False).clamp(0, 1)
    return (x - mean) / std


def _detect_model_family(model_name):
    """Infer the preprocessing family from the HF model id."""
    name = model_name.lower()
    if "siglip" in name:
        return "siglip2"  # siglip and siglip2 share the same normalization
    return "clip"


def _model_input_size(model_name):
    """Infer the vision input resolution from the model name.
    CLIP ViT-B/32: 224; SigLIP2 SO400M patch14-384: 384; SigLIP base-patch16-224: 224."""
    # Try to parse "-patchN-RES" suffix (common HF naming convention)
    import re
    m = re.search(r"patch\d+-(\d+)", model_name)
    if m:
        return int(m.group(1))
    # Fallback: CLIP ViT-B/32 is 224
    return 224


# ────────────────────────────────────────────
# Policy
# ────────────────────────────────────────────

class BernoulliPolicy:
    def __init__(self, n_bits, init_logit=0.0, device="cpu"):
        self.logits = torch.nn.Parameter(
            torch.full((n_bits,), init_logit, device=device, dtype=torch.float32)
        )

    def sample(self, batch_size):
        probs = torch.sigmoid(self.logits)
        dist = Bernoulli(probs)
        masks = dist.sample((batch_size,))  # (B, n_bits)
        log_probs = dist.log_prob(masks).sum(dim=-1)  # (B,)
        return masks, log_probs

    def get_probs(self):
        return torch.sigmoid(self.logits).detach()

    def entropy(self):
        return Bernoulli(logits=self.logits).entropy().sum()


# ────────────────────────────────────────────
# Diffusion Generator
# ────────────────────────────────────────────

class DiffusionGenerator:
    """Single-GPU FLUX generator that returns image tensors directly."""

    def __init__(self, device, source_prompt, target_prompt,
                 height, width, guidance_scale, seed, n_bits, steps):
        self.device = device
        self.guidance_scale = guidance_scale
        self.n_bits = n_bits
        self.steps = steps
        self.repeat_factor = steps // n_bits

        from diffusers import Flux2KleinPipeline

        print(f"Loading FLUX.2-klein-base-9B (BF16) on {device}...")
        # Load full pipeline for prompt encoding
        pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(device)

        # Encode prompts
        self.src_embeds, self.src_text_ids = pipe.encode_prompt(prompt=source_prompt, device=device)
        self.tgt_embeds, self.tgt_text_ids = pipe.encode_prompt(prompt=target_prompt, device=device)
        self.neg_embeds, self.neg_text_ids = pipe.encode_prompt(prompt="", device=device)
        print(f"  Embeddings: src={self.src_embeds.shape}, tgt={self.tgt_embeds.shape}")

        # Prepare shared noise on CPU then move
        num_channels = pipe.transformer.config.in_channels // 4
        generator = torch.Generator().manual_seed(seed)
        latents, _ = pipe.prepare_latents(
            1, num_channels, height, width,
            torch.bfloat16, torch.device("cpu"), generator, None,
        )
        self.shared_noise = latents.to(device=device, dtype=torch.bfloat16)
        print(f"  Noise shape: {self.shared_noise.shape}")

        # Prepare latent_ids
        num_channels_latents = pipe.transformer.config.in_channels // 4
        _, self.latent_ids = pipe.prepare_latents(
            1, num_channels_latents, height, width,
            torch.bfloat16, device, None, None,
        )
        self.latent_ids = self.latent_ids.to(device)

        # Prepare timesteps
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
        image_seq_len = self.shared_noise.shape[1]
        self.mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )
        self.retrieve_timesteps = retrieve_timesteps

        # Delete text encoder to free memory, keep transformer + vae + scheduler
        del pipe.text_encoder, pipe.tokenizer
        pipe.text_encoder = None
        pipe.tokenizer = None
        torch.cuda.empty_cache()

        # Compile transformer
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=False)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe

        vram = torch.cuda.memory_allocated(device) / 1e9
        print(f"  VRAM after init: {vram:.1f} GB")

    @torch.no_grad()
    def generate(self, masks):
        """Generate images from binary masks.

        Args:
            masks: (B, n_bits) float/bool tensor on any device.
        Returns:
            (B, 3, H, W) float32 tensor in [0, 1] on self.device.
        """
        masks_bool = masks.bool().to(self.device)
        cur_bs = masks_bool.shape[0]

        # Expand bits to steps
        step_masks = masks_bool.repeat_interleave(self.repeat_factor, dim=1)  # (B, steps)

        pipe = self.pipe
        device = self.device

        pipe.scheduler.set_begin_index(0)
        ts, _ = self.retrieve_timesteps(
            pipe.scheduler, self.steps, device, sigmas=None, mu=self.mu,
        )

        latents = self.shared_noise.expand(cur_bs, -1, -1).clone()
        latent_image_ids = self.latent_ids.expand(cur_bs, -1, -1)

        src_exp = self.src_embeds.expand(cur_bs, -1, -1)
        tgt_exp = self.tgt_embeds.expand(cur_bs, -1, -1)
        neg_pe = self.neg_embeds.expand(cur_bs, -1, -1)

        src_tid = self.src_text_ids
        if src_tid.dim() == 2:
            src_tid = src_tid.unsqueeze(0)
        src_tid = src_tid.expand(cur_bs, -1, -1)
        tgt_tid = self.tgt_text_ids
        if tgt_tid.dim() == 2:
            tgt_tid = tgt_tid.unsqueeze(0)
        tgt_tid = tgt_tid.expand(cur_bs, -1, -1)
        neg_tid = self.neg_text_ids
        if neg_tid.dim() == 2:
            neg_tid = neg_tid.unsqueeze(0)
        neg_tid = neg_tid.expand(cur_bs, -1, -1)

        for i, t in enumerate(ts):
            timestep = t.expand(cur_bs).to(latents.dtype)
            mask_i = step_masks[:, i]
            mask_3d = mask_i.view(cur_bs, 1, 1).expand_as(src_exp)

            pe = torch.where(mask_3d, tgt_exp, src_exp)
            mask_tid = mask_i.view(cur_bs, 1, 1).expand_as(src_tid)
            tid = torch.where(mask_tid, tgt_tid, src_tid)

            latent_model_input = latents.to(pipe.transformer.dtype)

            # Conditional pass
            noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=pe,
                txt_ids=tid,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, :latents.size(1)].clone()

            # Unconditional pass (CFG)
            neg_noise_pred = pipe.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=None,
                encoder_hidden_states=neg_pe,
                txt_ids=neg_tid,
                img_ids=latent_image_ids,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, :latents.size(1)]

            noise_pred = neg_noise_pred + self.guidance_scale * (noise_pred - neg_noise_pred)
            latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode
        lat_dec = pipe._unpack_latents_with_ids(latents, self.latent_ids.expand(cur_bs, -1, -1))
        bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_dec.device, lat_dec.dtype)
        bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
            lat_dec.device, lat_dec.dtype)
        lat_dec = lat_dec * bn_std + bn_mean
        lat_dec = pipe._unpatchify_latents(lat_dec)
        images_t = pipe.vae.decode(lat_dec, return_dict=False)[0]

        # Post-process to [0, 1] float32
        images_t = pipe.image_processor.postprocess(images_t, output_type="pt")
        return images_t.float()


# ────────────────────────────────────────────
# Reward Computer
# ────────────────────────────────────────────

class RewardComputer:
    """Computes reward from generated images using a vision-language model + SSIM on GPU.

    Supports CLIP (openai/clip-vit-*) and SigLIP/SigLIP2 (google/siglip*)."""

    def __init__(self, device, source_image, bg_mask, source_prompt, target_prompt, img_size,
                 vision_model="google/siglip2-so400m-patch14-384"):
        """
        Args:
            source_image: (1, 3, H, W) float32 tensor in [0,1] on device.
            bg_mask: (H, W) numpy array, 1=background, 0=foreground.
            vision_model: HF model id for CLIP-style or SigLIP-style model.
        """
        self.device = device
        self.img_size = img_size
        self.vision_model_name = vision_model
        self.model_family = _detect_model_family(vision_model)
        self.vis_input_size = _model_input_size(vision_model)

        # Background mask
        bg_mask_resized = Image.fromarray((bg_mask * 255).astype(np.uint8)).resize(
            (img_size, img_size), Image.NEAREST)
        bg_mask_np = (np.array(bg_mask_resized) > 127).astype(np.float32)
        self.bg_mask = torch.from_numpy(bg_mask_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        self.fg_mask = 1.0 - self.bg_mask
        self.bg_pixels = self.bg_mask.sum().item()
        self.gray = 128.0 / 255.0

        # Foreground bounding box for CLIP cropping
        fg_ys, fg_xs = np.where(bg_mask_np < 0.5)
        if len(fg_ys) == 0:
            # No foreground detected — use center crop as fallback
            print("  WARNING: mask has no foreground pixels, using center crop for fg CLIP")
            self.fg_y1, self.fg_x1 = img_size // 4, img_size // 4
            self.fg_y2, self.fg_x2 = img_size * 3 // 4, img_size * 3 // 4
        else:
            fg_y1, fg_y2 = int(fg_ys.min()), int(fg_ys.max()) + 1
            fg_x1, fg_x2 = int(fg_xs.min()), int(fg_xs.max()) + 1
            fg_h, fg_w = fg_y2 - fg_y1, fg_x2 - fg_x1
            fg_side = max(fg_h, fg_w)
            fg_cy, fg_cx = (fg_y1 + fg_y2) // 2, (fg_x1 + fg_x2) // 2
            self.fg_y1 = max(0, fg_cy - fg_side // 2)
            self.fg_x1 = max(0, fg_cx - fg_side // 2)
            self.fg_y2 = min(img_size, self.fg_y1 + fg_side)
            self.fg_x2 = min(img_size, self.fg_x1 + fg_side)

        # Source image on GPU
        self.source_dev = source_image.to(device)  # (1,3,H,W)

        # SSIM kernel
        self.ssim_kernel = _gaussian_kernel_2d(11, 1.5, 3, device=device, dtype=torch.float32)
        self.ssim_pad = 5

        # Vision-language model (CLIP or SigLIP / SigLIP2, unified via AutoModel/AutoProcessor)
        from transformers import AutoModel, AutoProcessor
        print(f"Loading vision-language model: {vision_model}")
        print(f"  family={self.model_family}  input_size={self.vis_input_size}")
        vl_model = AutoModel.from_pretrained(vision_model, torch_dtype=torch.float32).to(device)
        vl_model.eval()
        self.vl_model = vl_model
        self.vis_mean, self.vis_std = build_vision_transform(
            device, model_family=self.model_family)

        # Helper that uses the unified get_image_features / get_text_features API
        def _image_embed(img_batch_preprocessed):
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = vl_model.get_image_features(pixel_values=img_batch_preprocessed)
            return emb.float()

        self._image_embed = _image_embed

        # Pre-compute source background embedding
        source_bg = self.source_dev * self.bg_mask + self.gray * self.fg_mask
        source_bg_pp = vision_preprocess_gpu(
            source_bg, self.vis_mean, self.vis_std, self.vis_input_size)
        self.src_bg_emb = F.normalize(_image_embed(source_bg_pp), p=2, dim=-1)

        # Pre-compute text embeddings
        processor = AutoProcessor.from_pretrained(vision_model)

        def _get_text_emb(prompt):
            t = processor(text=[prompt], return_tensors="pt",
                          padding="max_length", truncation=True)
            t = {k: v.to(device) for k, v in t.items()}
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                emb = vl_model.get_text_features(**t)
            return emb.float()

        src_text_emb = _get_text_emb(source_prompt)
        tgt_text_emb = _get_text_emb(target_prompt)
        # Delta direction (legacy reward type)
        self.text_emb = F.normalize(tgt_text_emb - src_text_emb, p=2, dim=-1)
        # Individually normalized text embeddings (relative reward type — default)
        self.src_text_emb_norm = F.normalize(src_text_emb, p=2, dim=-1)
        self.tgt_text_emb_norm = F.normalize(tgt_text_emb, p=2, dim=-1)

        del processor
        torch.cuda.empty_cache()
        print(f"  Reward computer ready. BG pixels: {int(self.bg_pixels)}/{img_size**2}  "
              f"embed_dim={self.src_bg_emb.shape[-1]}")

    @torch.no_grad()
    def compute_rewards(self, images, alpha=0.5, reward_type="relative", clamp_fg=False):
        """Compute reward for a batch of images.

        Args:
            images: (B, 3, H, W) float32 in [0,1] on GPU.
            alpha: weight for bg_ssim vs fg_clip_score.
            reward_type: "relative" (fg·tgt - fg·src) or "delta" (fg · normalize(tgt-src)).
            clamp_fg: if True, clamp fg_clip_score below 0 (legacy behavior).
        Returns:
            (B,) reward tensor, bg_ssim array, fg_clip array.
        """
        B = images.shape[0]

        # bg_ssim
        smap = ssim_map(images, self.source_dev.expand(B, -1, -1, -1), self.ssim_kernel, self.ssim_pad)
        bg_ssim = (smap * self.bg_mask).sum(dim=(1, 2, 3)) / max(self.bg_pixels, 1.0)  # (B,)

        # Vision-language model: bg + fg in one forward pass
        imgs_bg = images * self.bg_mask + self.gray * self.fg_mask
        imgs_fg_crop = images[:, :, self.fg_y1:self.fg_y2, self.fg_x1:self.fg_x2]

        bg_pp = vision_preprocess_gpu(imgs_bg, self.vis_mean, self.vis_std, self.vis_input_size)
        fg_pp = vision_preprocess_gpu(imgs_fg_crop, self.vis_mean, self.vis_std, self.vis_input_size)
        combined = torch.cat([bg_pp, fg_pp], dim=0)  # (2B, 3, S, S)

        embs = F.normalize(self._image_embed(combined), p=2, dim=-1)
        fg_embs = embs[B:]
        if reward_type == "delta":
            fg_clip_score = (fg_embs * self.text_emb).sum(dim=-1)  # (B,)
        else:  # "relative"
            fg_to_tgt = (fg_embs * self.tgt_text_emb_norm).sum(dim=-1)
            fg_to_src = (fg_embs * self.src_text_emb_norm).sum(dim=-1)
            fg_clip_score = fg_to_tgt - fg_to_src  # (B,) in ~[-0.3, 0.3]

        fg_for_reward = fg_clip_score.clamp(min=0) if clamp_fg else fg_clip_score
        rewards = alpha * bg_ssim + (1.0 - alpha) * fg_for_reward
        return rewards, bg_ssim, fg_clip_score


# ────────────────────────────────────────────
# Segmentation bootstrap
# ────────────────────────────────────────────

def compute_segmentation(source_image_tensor, seg_prompt, device, dilate_px=10, threshold=0.5):
    """Run CLIPSeg on a source image tensor to get background mask.

    Args:
        source_image_tensor: (1, 3, H, W) float32 in [0,1].
    Returns:
        bg_mask: (H, W) numpy array, 1=background, 0=foreground.
    """
    from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

    # Convert tensor to PIL
    img_np = (source_image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_img = Image.fromarray(img_np)
    H, W = pil_img.size[1], pil_img.size[0]

    processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
    model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined").to(device)

    inputs = processor(text=[seg_prompt], images=[pil_img], return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
    upsampled = F.interpolate(
        logits.unsqueeze(1).float(), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze()

    prob = torch.sigmoid(upsampled).cpu().numpy()
    foreground = (prob > threshold).astype(np.uint8)

    if dilate_px > 0:
        from scipy.ndimage import binary_dilation
        struct = np.ones((dilate_px * 2 + 1, dilate_px * 2 + 1))
        foreground = binary_dilation(foreground, structure=struct).astype(np.uint8)

    del model, processor
    torch.cuda.empty_cache()

    bg_mask = 1 - foreground
    print(f"  Segmentation: foreground={100 * (1 - bg_mask.mean()):.1f}%, bg={100 * bg_mask.mean():.1f}%")
    return bg_mask


# ────────────────────────────────────────────
# Main training loop
# ────────────────────────────────────────────

def train_reinforce(args):
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    n_bits = args.n_bits
    steps = args.steps if args.steps else n_bits * 2

    # ── Phase 1: Initialize generator ──
    print("\n=== Phase 1: Initialize diffusion generator ===")
    generator = DiffusionGenerator(
        device=device,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        height=args.height, width=args.width,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        n_bits=n_bits, steps=steps,
    )

    # ── Phase 2: Generate source image and segment ──
    print("\n=== Phase 2: Source image & segmentation ===")
    if args.mask:
        print(f"  Using pre-computed mask: {args.mask}")
        bg_mask = np.load(args.mask)
        # Generate source image for reward computation
        zeros_mask = torch.zeros(1, n_bits, device=device)
        source_img = generator.generate(zeros_mask)  # (1, 3, H, W)
    else:
        zeros_mask = torch.zeros(1, n_bits, device=device)
        source_img = generator.generate(zeros_mask)
        seg_prompt = args.seg_prompt or args.source_prompt
        print(f"  Running CLIPSeg with prompt: '{seg_prompt}'")
        bg_mask = compute_segmentation(source_img, seg_prompt, device)

    # Save source image for reference
    src_pil = Image.fromarray(
        (source_img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    src_pil.save(os.path.join(args.output_dir, "source_b0.jpg"), quality=95)

    # Also generate and save target image (all-ones mask) as a reference
    ones_mask = torch.ones(1, n_bits, device=device)
    target_img = generator.generate(ones_mask)
    tgt_pil = Image.fromarray(
        (target_img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    target_b = (1 << n_bits) - 1
    tgt_pil.save(os.path.join(args.output_dir, f"target_b{target_b}.jpg"), quality=95)

    # Save prompts as plain text for downstream analysis
    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        f.write(f"source: {args.source_prompt}\n")
        f.write(f"target: {args.target_prompt}\n")
        f.write(f"seg: {args.seg_prompt or args.source_prompt}\n")
        f.write(f"reward_type: {args.reward_type}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"vision_model: {args.vision_model}\n")

    # ── Phase 3: Initialize reward computer ──
    print("\n=== Phase 3: Initialize reward computer ===")
    reward_computer = RewardComputer(
        device=device,
        source_image=source_img,
        bg_mask=bg_mask,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        img_size=args.height,
        vision_model=args.vision_model,
    )

    # ── Phase 4: REINFORCE training ──
    print(f"\n=== Phase 4: REINFORCE training ({args.num_episodes} episodes, BS={args.batch_size}) ===", flush=True)
    policy = BernoulliPolicy(n_bits, init_logit=0.0, device=device)
    optimizer = torch.optim.Adam([policy.logits], lr=args.lr)

    baseline = 0.0
    best_reward = -float("inf")
    best_mask = None
    best_image = None
    total_images = 0
    episodes_since_improvement = 0  # for plateau early stop

    # CSV log
    log_path = os.path.join(args.output_dir, "reinforce_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    header = ["episode", "mean_reward", "best_reward_ever", "mean_bg_ssim", "mean_fg_clip",
              "entropy", "baseline"] + [f"prob_{i}" for i in range(n_bits)]
    log_writer.writerow(header)

    t_start = time.perf_counter()

    for ep in range(args.num_episodes):
        # Sample masks
        masks, log_probs = policy.sample(args.batch_size)  # (B, n_bits), (B,)

        # Generate images and compute rewards
        with torch.no_grad():
            images = generator.generate(masks)  # (B, 3, H, W)
            rewards, bg_ssim_vals, fg_clip_vals = reward_computer.compute_rewards(
                images, args.alpha, reward_type=args.reward_type, clamp_fg=args.clamp_fg)

        total_images += args.batch_size
        mean_reward = rewards.mean().item()

        # Update baseline (EMA)
        baseline = args.baseline_ema * baseline + (1.0 - args.baseline_ema) * mean_reward

        # Compute advantage
        advantage = rewards - baseline
        if args.normalize_advantages and advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

        # REINFORCE loss
        policy_loss = -(advantage.detach() * log_probs).mean()
        entropy = policy.entropy()
        loss = policy_loss - args.entropy_coeff * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track best
        batch_best_idx = rewards.argmax().item()
        if rewards[batch_best_idx] > best_reward:
            best_reward = rewards[batch_best_idx].item()
            best_mask = masks[batch_best_idx].detach().clone()
            best_image = images[batch_best_idx].detach().clone()
            episodes_since_improvement = 0
        else:
            episodes_since_improvement += 1

        # Log
        probs = policy.get_probs().cpu().numpy()
        log_writer.writerow([
            ep, f"{mean_reward:.6f}", f"{best_reward:.6f}",
            f"{bg_ssim_vals.mean().item():.6f}", f"{fg_clip_vals.mean().item():.6f}",
            f"{entropy.item():.4f}", f"{baseline:.6f}",
        ] + [f"{p:.4f}" for p in probs])
        log_file.flush()

        if ep % args.log_interval == 0:
            elapsed = time.perf_counter() - t_start
            probs_str = " ".join(f"{p:.2f}" for p in probs)
            print(f"  [{ep:4d}/{args.num_episodes}] "
                  f"R={mean_reward:.4f} best={best_reward:.4f} "
                  f"bg={bg_ssim_vals.mean():.3f} fg={fg_clip_vals.mean():.3f} "
                  f"H={entropy.item():.2f} "
                  f"({total_images} imgs, {elapsed:.0f}s)", flush=True)
            print(f"         probs=[{probs_str}]", flush=True)

        # Early stopping conditions (disabled when patience/threshold are 0)
        if args.entropy_stop > 0 and entropy.item() < args.entropy_stop:
            print(f"  Early stop at episode {ep}: entropy={entropy.item():.3f} < {args.entropy_stop}", flush=True)
            break
        if args.plateau_patience > 0 and episodes_since_improvement >= args.plateau_patience:
            print(f"  Early stop at episode {ep}: no improvement in {episodes_since_improvement} episodes "
                  f"(best_reward={best_reward:.4f})", flush=True)
            break

    log_file.close()
    elapsed = time.perf_counter() - t_start
    print(f"\nTraining done: {total_images} images in {elapsed:.1f}s ({total_images/elapsed:.1f} img/s)")

    # ── Phase 5: Generate top-K results ──
    print(f"\n=== Phase 5: Generate top-{args.top_k} images ===")
    final_probs = policy.get_probs()
    print(f"  Learned probabilities: {' '.join(f'{p:.3f}' for p in final_probs.cpu().numpy())}")

    # Greedy mask (threshold at 0.5)
    greedy_mask = (final_probs > 0.5).float().unsqueeze(0)  # (1, n_bits)

    # Sample additional masks from learned distribution
    if args.top_k > 1:
        extra_masks, _ = policy.sample(args.top_k - 1)
        all_masks = torch.cat([greedy_mask, extra_masks], dim=0)
    else:
        all_masks = greedy_mask

    # Add the best mask found during training
    if best_mask is not None:
        all_masks = torch.cat([best_mask.unsqueeze(0), all_masks], dim=0)

    with torch.no_grad():
        all_images = generator.generate(all_masks)
        all_rewards, all_bg, all_fg = reward_computer.compute_rewards(
            all_images, args.alpha, reward_type=args.reward_type, clamp_fg=args.clamp_fg)

    # Sort by reward
    order = all_rewards.argsort(descending=True)
    masks_np = all_masks[order].cpu().numpy().astype(int)
    mask_ints = mask_to_int_batch(masks_np)

    for i in range(len(order)):
        idx = order[i].item()
        r = all_rewards[idx].item()
        b_int = mask_ints[i]
        img_np = (all_images[idx].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        fname = f"reinforce_top{i}_r{r:.4f}_b{b_int}.jpg"
        pil_img.save(os.path.join(args.output_dir, fname), quality=95)
        bits_str = "".join(str(b) for b in masks_np[i])
        print(f"  {fname}  bg={all_bg[idx]:.4f} fg={all_fg[idx]:.4f} bits={bits_str}")

    # Save checkpoint
    ckpt_path = os.path.join(args.output_dir, "reinforce_result.pt")
    torch.save({
        "logits": policy.logits.data.cpu(),
        "probs": final_probs.cpu(),
        "best_mask": best_mask.cpu() if best_mask is not None else None,
        "best_reward": best_reward,
        "total_images": total_images,
        "n_bits": n_bits,
        "steps": steps,
        "args": vars(args),
    }, ckpt_path)
    print(f"\nCheckpoint saved to {ckpt_path}")
    print(f"Log saved to {log_path}")
    print(f"Total images evaluated: {total_images} (vs {1 << n_bits} exhaustive)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="REINFORCE binary path search for FLUX")
    # Generation
    p.add_argument("--source_prompt", required=True)
    p.add_argument("--target_prompt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_bits", type=int, default=14)
    p.add_argument("--steps", type=int, default=None, help="Total diffusion steps (default: n_bits * 2)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8)
    # RL
    p.add_argument("--num_episodes", type=int, default=300,
                   help="Hard cap. Empirically runs early-stop around ep 200-300 so 300 is a safe ceiling.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--alpha", type=float, default=0.3, help="Reward weight: alpha*bg_ssim + (1-alpha)*fg_clip")
    p.add_argument("--reward_type", choices=["relative", "delta"], default="relative",
                   help="relative: fg·tgt - fg·src (new default); delta: fg · normalize(tgt-src) (legacy)")
    p.add_argument("--clamp_fg", action="store_true",
                   help="Clamp fg_clip_score below zero to 0 (legacy behavior)")
    p.add_argument("--baseline_ema", type=float, default=0.9)
    p.add_argument("--entropy_coeff", type=float, default=0.05)
    p.add_argument("--normalize_advantages", action="store_true", default=True,
                   help="Standardize advantages per batch (default on; use --no-normalize_advantages to disable)")
    p.add_argument("--no-normalize_advantages", dest="normalize_advantages", action="store_false")
    # Early-stop criteria
    p.add_argument("--entropy_stop", type=float, default=0.5,
                   help="Stop when policy entropy drops below this value. Set 0 to disable.")
    p.add_argument("--plateau_patience", type=int, default=100,
                   help="Stop if best_reward has not improved for this many episodes. Set 0 to disable.")
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=10)
    # Segmentation
    p.add_argument("--seg_prompt", default=None, help="Prompt for CLIPSeg (defaults to source_prompt)")
    p.add_argument("--mask", default=None, help="Pre-computed background mask (.npy)")
    # Vision-language model for reward
    p.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384",
                   help="HF id for reward VLM. Supports CLIP (openai/clip-vit-*) and "
                        "SigLIP/SigLIP2 (google/siglip*). Default: SigLIP2 SO400M @ 384.")

    args = p.parse_args()
    train_reinforce(args)
