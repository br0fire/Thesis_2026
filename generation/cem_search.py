"""Cross-Entropy Method for binary path search.

Alternative to REINFORCE: samples a batch of masks, keeps top-K elite by reward,
fits new Bernoulli probs from the elite's bit frequencies (smoothed with previous
probs). Generally more sample-efficient than REINFORCE on discrete spaces for
the same budget.

Usage (same CLI as reinforce_search.py):
  python generation/cem_search.py \\
      --source_prompt ... --target_prompt ... --seg_prompt ... \\
      --output_dir ... --gpu 0 \\
      --n_bits 14 --batch_size 8 --num_iterations 20 \\
      --elite_frac 0.25 --smoothing 0.3
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Reuse components from reinforce_search.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reinforce_search import DiffusionGenerator, RewardComputer, compute_segmentation, mask_to_int_batch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source_prompt", required=True)
    p.add_argument("--target_prompt", required=True)
    p.add_argument("--seg_prompt", default=None)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n_bits", type=int, default=14)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Number of masks sampled per iteration")
    p.add_argument("--num_iterations", type=int, default=20,
                   help="Number of CEM iterations (total images = batch_size × num_iterations)")
    p.add_argument("--elite_frac", type=float, default=0.25,
                   help="Fraction of batch kept as elite (default 0.25 → top 25%)")
    p.add_argument("--smoothing", type=float, default=0.3,
                   help="Smoothing factor for probs update: "
                        "new_p = (1-s)*empirical_p + s*old_p. "
                        "s=0.0 → pure empirical, s=1.0 → no update")
    p.add_argument("--prob_clip", type=float, default=0.05,
                   help="Clip probs to [clip, 1-clip] to prevent collapse")
    p.add_argument("--top_k", type=int, default=10, help="Final top-K images to save")
    p.add_argument("--log_interval", type=int, default=1)
    p.add_argument("--mask", default=None, help="Pre-computed background mask")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--vision_model", default="google/siglip2-so400m-patch14-384")
    p.add_argument("--seg_method", choices=["sam3", "gdino_sam", "clipseg"], default="sam3")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    os.makedirs(args.output_dir, exist_ok=True)

    n_bits = args.n_bits
    steps = args.steps if args.steps else n_bits * 2

    # ── Phase 1: generator ──
    print("=== Phase 1: load diffusion generator ===", flush=True)
    generator = DiffusionGenerator(
        device=device, source_prompt=args.source_prompt, target_prompt=args.target_prompt,
        height=args.height, width=args.width, guidance_scale=args.guidance_scale,
        seed=args.seed, n_bits=n_bits, steps=steps,
    )

    # ── Phase 2: source + seg ──
    print("\n=== Phase 2: source image + segmentation ===", flush=True)
    zeros_mask = torch.zeros(1, n_bits, device=device)
    source_img = generator.generate(zeros_mask)
    if args.mask:
        bg_mask = np.load(args.mask)
    else:
        seg_prompt = args.seg_prompt or args.source_prompt
        bg_mask = compute_segmentation(source_img, seg_prompt, device, method=args.seg_method)
    src_pil = Image.fromarray(
        (source_img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
    src_pil.save(os.path.join(args.output_dir, "source_b0.jpg"), quality=95)

    # Save prompts
    with open(os.path.join(args.output_dir, "prompts.txt"), "w") as f:
        f.write(f"source: {args.source_prompt}\n")
        f.write(f"target: {args.target_prompt}\n")
        f.write(f"seg: {args.seg_prompt or args.source_prompt}\n")
        f.write(f"alpha: {args.alpha}\n")
        f.write(f"method: cem\n")
        f.write(f"vision_model: {args.vision_model}\n")

    # ── Phase 3: reward computer ──
    print("\n=== Phase 3: reward computer ===", flush=True)
    reward_computer = RewardComputer(
        device=device, source_image=source_img, bg_mask=bg_mask,
        source_prompt=args.source_prompt, target_prompt=args.target_prompt,
        img_size=args.height, vision_model=args.vision_model,
    )

    # ── Phase 4: CEM loop ──
    print("\n=== Phase 4: CEM ===", flush=True)
    probs = torch.full((n_bits,), 0.5, device=device)
    elite_k = max(1, int(args.batch_size * args.elite_frac))
    total_images = 0
    best_reward = -float("inf")
    best_mask = None
    best_image = None

    log_path = os.path.join(args.output_dir, "cem_log.csv")
    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    header = (["iteration", "mean_reward", "best_reward_ever", "elite_mean_reward",
               "mean_bg_ssim", "mean_fg_clip", "entropy"]
              + [f"prob_{i}" for i in range(n_bits)])
    log_writer.writerow(header)

    t0 = time.perf_counter()
    for it in range(args.num_iterations):
        # Sample batch
        dist = torch.distributions.Bernoulli(probs.unsqueeze(0).expand(args.batch_size, -1))
        masks = dist.sample()

        with torch.no_grad():
            images = generator.generate(masks)
            rewards, bg_ssim_vals, fg_clip_vals = reward_computer.compute_rewards(
                images, args.alpha)
        total_images += args.batch_size

        # Track best
        batch_best_idx = rewards.argmax().item()
        if rewards[batch_best_idx] > best_reward:
            best_reward = rewards[batch_best_idx].item()
            best_mask = masks[batch_best_idx].detach().clone()
            best_image = images[batch_best_idx].detach().clone()

        # Elite selection
        sorted_idx = torch.argsort(rewards, descending=True)
        elite_idx = sorted_idx[:elite_k]
        elite_masks = masks[elite_idx]
        elite_rewards = rewards[elite_idx]
        empirical_p = elite_masks.mean(dim=0)

        # Smoothed update
        new_probs = (1.0 - args.smoothing) * empirical_p + args.smoothing * probs
        new_probs = new_probs.clamp(args.prob_clip, 1.0 - args.prob_clip)
        probs = new_probs

        # Log
        eps = 1e-9
        entropy = -(probs * (probs + eps).log() + (1 - probs) * (1 - probs + eps).log()).sum().item()
        mean_r = rewards.mean().item()
        elite_mean = elite_rewards.mean().item()
        probs_np = probs.cpu().numpy()
        log_writer.writerow([
            it, f"{mean_r:.6f}", f"{best_reward:.6f}", f"{elite_mean:.6f}",
            f"{bg_ssim_vals.mean().item():.6f}", f"{fg_clip_vals.mean().item():.6f}",
            f"{entropy:.4f}",
        ] + [f"{p:.4f}" for p in probs_np])
        log_file.flush()

        if it % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            probs_str = " ".join(f"{p:.2f}" for p in probs_np)
            print(f"  [it {it:3d}/{args.num_iterations}]  "
                  f"R={mean_r:.4f}  elite={elite_mean:.4f}  best={best_reward:.4f}  "
                  f"H={entropy:.2f}  ({total_images} imgs, {elapsed:.0f}s)", flush=True)
            print(f"         probs=[{probs_str}]", flush=True)

    log_file.close()

    # ── Phase 5: final top-K generation ──
    print("\n=== Phase 5: final top-K ===", flush=True)
    final_masks = [best_mask] + [
        (probs > 0.5).float(),  # greedy
    ]
    # Sample (top_k - 2) from final policy
    dist = torch.distributions.Bernoulli(probs.unsqueeze(0).expand(args.top_k - 2, -1))
    final_masks.extend(list(dist.sample()))
    final_masks = torch.stack([m.to(device) for m in final_masks])

    with torch.no_grad():
        final_images = generator.generate(final_masks)
        final_rewards, _, _ = reward_computer.compute_rewards(final_images, args.alpha)

    # Save top-K
    mask_ints = mask_to_int_batch(final_masks.cpu().numpy().astype(np.int64))
    order = torch.argsort(final_rewards, descending=True).cpu().numpy()
    for rank, i in enumerate(order[:args.top_k]):
        r = final_rewards[i].item()
        bint = mask_ints[i]
        img_np = (final_images[i].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img_np).save(
            os.path.join(args.output_dir, f"cem_top{rank}_r{r:.4f}_b{bint}.jpg"), quality=95)

    # Save checkpoint
    result = {
        "method": "cem",
        "probs": probs.cpu(),
        "best_mask": best_mask.cpu(),
        "best_reward": best_reward,
        "total_images": total_images,
        "num_iterations": args.num_iterations,
        "args": vars(args),
    }
    torch.save(result, os.path.join(args.output_dir, "cem_result.pt"))

    total_time = time.perf_counter() - t0
    print(f"\n=== Done. best_reward={best_reward:.4f}  total={total_time:.0f}s  "
          f"images={total_images} ===", flush=True)


if __name__ == "__main__":
    main()
