"""
FLUX.2-klein-base-9B binary path generation with batching + CFG.

No null-text inversion — start from shared random noise.
Each of 14 bits selects source or target prompt at a pair of diffusion steps (28 total).
Multi-GPU via spawn. Text encoder offloaded after encoding. CFG with empty negative prompt.
"""
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

MODEL_ID = "black-forest-labs/FLUX.2-klein-base-9B"
N_BITS = 14  # default, overridden by --n_bits
STEPS = 28   # default, overridden by --steps
BATCH_SIZE = 4


def mask_to_int_batch(masks):
    bits = np.arange(masks.shape[1] - 1, -1, -1)
    return (masks << bits).sum(axis=1)


def worker_fn(rank, gpu_ids, masks_chunk, global_indices_chunk,
              shared_noise_cpu, src_embeds_cpu, src_text_ids_cpu,
              tgt_embeds_cpu, tgt_text_ids_cpu,
              neg_embeds_cpu, neg_text_ids_cpu,
              pipe_config, progress_queue, output_dir):
    try:
        gpu_id = gpu_ids[rank]
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)

        from diffusers import Flux2KleinPipeline

        print(f"[GPU {gpu_id}] Loading FLUX.2-klein-base-9B (BF16, no text encoder)...")
        pipe = Flux2KleinPipeline.from_pretrained(
            MODEL_ID,
            text_encoder=None,
            tokenizer=None,
            torch_dtype=torch.bfloat16,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=False)

        vram = torch.cuda.memory_allocated(device) / 1e9
        print(f"[GPU {gpu_id}] VRAM after load: {vram:.1f} GB")

        # Move pre-computed embeddings to GPU
        src_embeds = src_embeds_cpu.to(device=device, dtype=torch.bfloat16)
        src_text_ids = src_text_ids_cpu.to(device=device)
        tgt_embeds = tgt_embeds_cpu.to(device=device, dtype=torch.bfloat16)
        tgt_text_ids = tgt_text_ids_cpu.to(device=device)
        neg_embeds = neg_embeds_cpu.to(device=device, dtype=torch.bfloat16)
        neg_text_ids = neg_text_ids_cpu.to(device=device)

        height = pipe_config["height"]
        width = pipe_config["width"]
        batch_size = pipe_config.get("batch_size", BATCH_SIZE)
        guidance_scale = pipe_config.get("guidance_scale", 4.0)
        n_bits = pipe_config.get("n_bits", N_BITS)
        steps = pipe_config.get("steps", STEPS)

        shared_noise_single = shared_noise_cpu.to(device=device, dtype=torch.bfloat16)

        # Prepare latent_ids
        num_channels_latents = pipe.transformer.config.in_channels // 4
        _, latent_ids_single = pipe.prepare_latents(
            1, num_channels_latents, height, width,
            torch.bfloat16, device, None, None,
        )
        latent_ids_single = latent_ids_single.to(device)

        # Prepare timesteps
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift, retrieve_timesteps
        image_seq_len = shared_noise_single.shape[1]
        mu = calculate_shift(
            image_seq_len,
            pipe.scheduler.config.get("base_image_seq_len", 256),
            pipe.scheduler.config.get("max_image_seq_len", 4096),
            pipe.scheduler.config.get("base_shift", 0.5),
            pipe.scheduler.config.get("max_shift", 1.15),
        )

        num_samples = len(masks_chunk)
        repeat_factor = steps // n_bits

        def _save_image(pil_img, fpath):
            pil_img.save(fpath, quality=95)

        save_executor = ThreadPoolExecutor(max_workers=8)
        save_futures = []

        import time
        t_start = time.perf_counter()
        print(f"[GPU {gpu_id}] Processing {num_samples} paths (BS={batch_size}, CFG={guidance_scale})...")

        for batch_start in range(0, num_samples, batch_size):
            for fut in save_futures:
                fut.result()
            save_futures = []

            batch_end = min(batch_start + batch_size, num_samples)
            cur_bs = batch_end - batch_start

            batch_masks = masks_chunk[batch_start:batch_end]
            batch_indices = global_indices_chunk[batch_start:batch_end]

            step_masks = np.repeat(batch_masks, repeat_factor, axis=1)
            step_masks_t = torch.tensor(step_masks, dtype=torch.bool, device=device)

            pipe.scheduler.set_begin_index(0)
            ts, _ = retrieve_timesteps(
                pipe.scheduler, steps, device, sigmas=None, mu=mu,
            )

            latents = shared_noise_single.expand(cur_bs, -1, -1).clone()
            latent_image_ids = latent_ids_single.expand(cur_bs, -1, -1)

            # Negative embeds expanded for batch
            neg_pe = neg_embeds.expand(cur_bs, -1, -1)
            neg_tid = neg_text_ids.expand(cur_bs, -1, -1) if neg_text_ids.dim() == 3 else neg_text_ids.unsqueeze(0).expand(cur_bs, -1, -1)

            for i, t in enumerate(ts):
                timestep = t.expand(cur_bs).to(latents.dtype)

                # Build conditional prompt embeds based on mask
                mask_i = step_masks_t[:, i]
                src_exp = src_embeds.expand(cur_bs, -1, -1)
                tgt_exp = tgt_embeds.expand(cur_bs, -1, -1)
                mask_3d = mask_i.view(cur_bs, 1, 1).expand_as(src_exp)
                pe = torch.where(mask_3d, tgt_exp, src_exp)

                src_tid_exp = src_text_ids.expand(cur_bs, -1, -1) if src_text_ids.dim() == 3 else src_text_ids.unsqueeze(0).expand(cur_bs, -1, -1)
                tgt_tid_exp = tgt_text_ids.expand(cur_bs, -1, -1) if tgt_text_ids.dim() == 3 else tgt_text_ids.unsqueeze(0).expand(cur_bs, -1, -1)
                mask_tid = mask_i.view(cur_bs, 1, 1).expand_as(src_tid_exp)
                tid = torch.where(mask_tid, tgt_tid_exp, src_tid_exp)

                latent_model_input = latents.to(pipe.transformer.dtype)

                with torch.no_grad():
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

                    # CFG combine
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Decode
            with torch.no_grad():
                lat_dec = pipe._unpack_latents_with_ids(latents, latent_ids_single.expand(cur_bs, -1, -1))
                bn_mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(lat_dec.device, lat_dec.dtype)
                bn_std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps).to(
                    lat_dec.device, lat_dec.dtype)
                lat_dec = lat_dec * bn_std + bn_mean
                lat_dec = pipe._unpatchify_latents(lat_dec)
                images_t = pipe.vae.decode(lat_dec, return_dict=False)[0]
                images_np = pipe.image_processor.postprocess(images_t, output_type="np")

            mask_ints = mask_to_int_batch(batch_masks)
            for k in range(cur_bs):
                img_np = (images_np[k] * 255).clip(0, 255).astype(np.uint8)
                pil_img = Image.fromarray(img_np)
                fname = f"path_{batch_indices[k]:05d}_b{mask_ints[k]}.jpg"
                out_path = os.path.join(output_dir, fname)
                save_futures.append(save_executor.submit(_save_image, pil_img, out_path))

            if progress_queue is not None:
                progress_queue.put(cur_bs)

            if batch_start > 0 and batch_start % (batch_size * 20) == 0:
                elapsed = time.perf_counter() - t_start
                done = batch_start + cur_bs
                print(f"[GPU {gpu_id}] {done}/{num_samples} ({done/elapsed:.2f} img/s)")

        for fut in save_futures:
            fut.result()
        save_executor.shutdown(wait=True)

        elapsed = time.perf_counter() - t_start
        print(f"[GPU {gpu_id}] Done: {num_samples} images in {elapsed:.1f}s ({num_samples/elapsed:.2f} img/s)")

    except Exception as e:
        print(f"[GPU {gpu_id}] ERROR: {e}")
        import traceback
        traceback.print_exc()


def run_flux_search(source_prompt, target_prompt, num_paths_limit, gpu_ids, output_dir,
                    height=1024, width=1024, guidance_scale=4.0, seed=42, batch_size=BATCH_SIZE):
    total_possible = 1 << N_BITS
    num_paths = min(num_paths_limit, total_possible)

    print(f"FLUX.2-klein-base-9B binary path generation: {N_BITS} bits × {STEPS//N_BITS} repeat = {STEPS} steps")
    print(f"Generating {num_paths} paths (out of {total_possible} possible), BS={batch_size}, CFG={guidance_scale}")

    # Build masks
    if num_paths >= total_possible:
        all_ints = np.arange(total_possible)
        masks = ((all_ints[:, None] >> np.arange(N_BITS - 1, -1, -1)) & 1).astype(int)
    else:
        path_source = np.zeros((1, N_BITS), dtype=int)
        path_target = np.ones((1, N_BITS), dtype=int)
        num_random = max(0, num_paths - 2)
        rng = np.random.default_rng(seed)
        chosen = rng.choice(total_possible - 2, size=num_random, replace=False) + 1
        random_masks = ((chosen[:, None] >> np.arange(N_BITS - 1, -1, -1)) & 1).astype(int)
        masks = np.vstack([path_source, path_target, random_masks])

    global_indices = np.arange(len(masks))
    os.makedirs(output_dir, exist_ok=True)

    # --- Phase 1: encode prompts on single GPU ---
    print("\n--- Phase 1: Encoding prompts (single GPU) ---")
    from diffusers import Flux2KleinPipeline

    encode_device = torch.device(f"cuda:{gpu_ids[0]}")
    torch.cuda.set_device(encode_device)

    pipe = Flux2KleinPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to(encode_device)

    src_embeds, src_text_ids = pipe.encode_prompt(prompt=source_prompt, device=encode_device)
    tgt_embeds, tgt_text_ids = pipe.encode_prompt(prompt=target_prompt, device=encode_device)
    neg_embeds, neg_text_ids = pipe.encode_prompt(prompt="", device=encode_device)
    print(f"  Source embeds: {src_embeds.shape}, Target embeds: {tgt_embeds.shape}")
    print(f"  Negative embeds: {neg_embeds.shape}")

    # Prepare shared noise
    num_channels = pipe.transformer.config.in_channels // 4
    generator = torch.Generator().manual_seed(seed)
    latents, _ = pipe.prepare_latents(
        1, num_channels, height, width,
        torch.bfloat16, torch.device("cpu"), generator, None,
    )
    shared_noise_cpu = latents.clone()
    print(f"  Noise shape: {shared_noise_cpu.shape}")

    # Move to CPU (detach for cross-process)
    src_embeds_cpu = src_embeds.detach().cpu()
    src_text_ids_cpu = src_text_ids.detach().cpu()
    tgt_embeds_cpu = tgt_embeds.detach().cpu()
    tgt_text_ids_cpu = tgt_text_ids.detach().cpu()
    neg_embeds_cpu = neg_embeds.detach().cpu()
    neg_text_ids_cpu = neg_text_ids.detach().cpu()

    del pipe
    torch.cuda.empty_cache()

    # --- Phase 2: distributed generation ---
    print(f"\n--- Phase 2: Distributed generation ({len(gpu_ids)} GPUs, CFG={guidance_scale}) ---")

    pipe_config = {
        "height": height,
        "width": width,
        "guidance_scale": guidance_scale,
        "batch_size": batch_size,
        "n_bits": N_BITS,
        "steps": STEPS,
    }

    num_gpus = len(gpu_ids)
    chunks_masks = np.array_split(masks, num_gpus)
    chunks_indices = np.array_split(global_indices, num_gpus)

    for rank, gid in enumerate(gpu_ids):
        print(f"  GPU {gid}: {len(chunks_masks[rank])} paths")

    ctx = mp.get_context('spawn')
    progress_queue = ctx.Queue()
    processes = []

    for rank in range(num_gpus):
        p = ctx.Process(
            target=worker_fn,
            args=(rank, gpu_ids, chunks_masks[rank], chunks_indices[rank],
                  shared_noise_cpu, src_embeds_cpu, src_text_ids_cpu,
                  tgt_embeds_cpu, tgt_text_ids_cpu,
                  neg_embeds_cpu, neg_text_ids_cpu,
                  pipe_config, progress_queue, output_dir),
        )
        p.daemon = True
        p.start()
        processes.append(p)

    total_paths = len(masks)
    processed = 0
    with tqdm(total=total_paths, unit="img", desc="FLUX Paths") as pbar:
        while processed < total_paths:
            try:
                if not any(p.is_alive() for p in processes) and progress_queue.empty():
                    break
                n = progress_queue.get(timeout=5)
                processed += n
                pbar.update(n)
            except:
                pass

    for p in processes:
        p.join()

    print("Generation complete.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source_prompt", required=True)
    p.add_argument("--target_prompt", required=True)
    p.add_argument("--num_paths", type=int, default=16384)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--guidance_scale", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_bits", type=int, default=14)
    p.add_argument("--steps", type=int, default=None, help="Total diffusion steps (default: n_bits * 2)")
    args = p.parse_args()

    N_BITS = args.n_bits
    STEPS = args.steps if args.steps else N_BITS * 2

    gpu_ids = [int(x) for x in args.gpus.split(",")]
    run_flux_search(
        args.source_prompt, args.target_prompt,
        args.num_paths, gpu_ids, args.output_dir,
        args.height, args.width, args.guidance_scale, args.seed, args.batch_size,
    )
