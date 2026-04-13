import os
import numpy as np
import torch
import torch.nn.functional as nnf
import torch.multiprocessing as mp
from PIL import Image
from torch.optim.adam import Adam
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore")

# A100 optimizations: TF32 for FP32 ops, cuDNN autotuning
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DDIMScheduler

# --- 1. Вспомогательные функции ---

def mask_to_int_batch(masks):
    """Vectorized: convert array of binary masks to integers (MSB-first)."""
    bits = np.arange(masks.shape[1] - 1, -1, -1)
    return (masks << bits).sum(axis=1)

# --- 2. Pipeline (Оптимизированный) ---

class NullTextPipeline(StableDiffusionPipeline):
    # Стандартные методы оставляем без изменений для краткости
    def get_noise_pred(self, latents, t, context):
        latents_input = torch.cat([latents] * 2)
        guidance_scale = 7.5
        noise_pred = self.unet(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
        latents = self.prev_step(noise_pred, t, latents)
        return latents

    @torch.no_grad()
    def image2latent(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), Image.LANCZOS)
        image = np.array(image)
        image = torch.from_numpy(image).float() / 127.5 - 1
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = image.to(device=self.device, dtype=self.vae.dtype)
        latents = self.vae.encode(image)["latent_dist"].mean * 0.18215
        return latents

    def null_optimization(self, latents, context, num_inner_steps, epsilon):
        uncond_embeddings, cond_embeddings = context.chunk(2)
        uncond_embeddings_list = []
        latent_cur = latents[-1]
        bar = tqdm(total=num_inner_steps * self.num_inference_steps)
        for i in range(self.num_inference_steps):
            uncond_embeddings = uncond_embeddings.clone().detach()
            uncond_embeddings.requires_grad = True
            optimizer = Adam([uncond_embeddings], lr=1e-2 * (1.0 - i / 100.0))
            latent_prev = latents[len(latents) - i - 2]
            t = self.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.get_noise_pred_single(latent_cur, t, cond_embeddings)
            for j in range(num_inner_steps):
                noise_pred_uncond = self.get_noise_pred_single(latent_cur, t, uncond_embeddings)
                noise_pred = noise_pred_uncond + 7.5 * (noise_pred_cond - noise_pred_uncond)
                latents_prev_rec = self.prev_step(noise_pred, t, latent_cur)
                loss = nnf.mse_loss(latents_prev_rec, latent_prev)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss_item = loss.item()
                bar.update()
                if loss_item < epsilon + i * 2e-5:
                    break
            for j in range(j + 1, num_inner_steps):
                bar.update()
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())
            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latent_cur = self.get_noise_pred(latent_cur, t, context)
        bar.close()
        return uncond_embeddings_list

    def invert(self, image_path: str, prompt: str, num_inner_steps=10, early_stop_epsilon=1e-6, num_inference_steps=50):
        self.num_inference_steps = num_inference_steps
        context = self.get_context(prompt)
        latent = self.image2latent(image_path)
        ddim_latents = self.ddim_inversion_loop(latent, context)
        cache_path = f"{image_path}.steps{num_inference_steps}.pt"
        if os.path.exists(cache_path):
            uncond_embeddings = torch.load(cache_path)
        else:
            uncond_embeddings = self.null_optimization(ddim_latents, context, num_inner_steps, early_stop_epsilon)
            uncond_embeddings = torch.stack(uncond_embeddings, 0)
            torch.save(uncond_embeddings, cache_path)
        return ddim_latents[-1], uncond_embeddings
    
    def get_noise_pred_single(self, latents, t, context):
        noise_pred = self.unet(latents, t, encoder_hidden_states=context)["sample"]
        return noise_pred

    def prev_step(self, model_output, timestep, sample):
        prev_timestep = timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
        alpha_prod_t_prev = (self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
        prev_sample = alpha_prod_t_prev**0.5 * pred_original_sample + pred_sample_direction
        return prev_sample

    def next_step(self, model_output, timestep, sample):
        timestep, next_timestep = (min(timestep - self.scheduler.config.num_train_timesteps // self.num_inference_steps, 999), timestep)
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t**0.5 * model_output) / alpha_prod_t**0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next**0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_context(self, prompt):
        uncond_input = self.tokenizer([""], padding="max_length", max_length=self.tokenizer.model_max_length, return_tensors="pt")
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        text_input = self.tokenizer([prompt], padding="max_length", max_length=self.tokenizer.model_max_length, truncation=True, return_tensors="pt")
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        return torch.cat([uncond_embeddings, text_embeddings])

    @torch.no_grad()
    def ddim_inversion_loop(self, latent, context):
        self.scheduler.set_timesteps(self.num_inference_steps)
        _, cond_embeddings = context.chunk(2)
        all_latent = [latent]
        latent = latent.clone().detach()
        for i in range(0, self.num_inference_steps):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
            latent = self.next_step(noise_pred, t, latent)
            all_latent.append(latent)
        return all_latent

    @torch.inference_mode()
    def sample_final_only(
        self,
        source_prompt_embeds: torch.Tensor,
        target_prompt_embeds: torch.Tensor,
        uncond_embeddings: torch.Tensor,
        inverted_latent: torch.Tensor,
        path_masks: torch.Tensor,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 50,
    ):
        device = self._execution_device
        dtype = self.unet.dtype

        batch_size, steps = path_masks.shape

        # Подготовка латентов (channels_last for tensor core optimization)
        latents = inverted_latent.to(device=device, dtype=dtype).expand(batch_size, -1, -1, -1).contiguous()
        latents = latents.to(memory_format=torch.channels_last)
        uncond_embeddings = uncond_embeddings.to(device=device, dtype=dtype)
        source_prompt_embeds = source_prompt_embeds.to(device=device, dtype=dtype)
        target_prompt_embeds = target_prompt_embeds.to(device=device, dtype=dtype)
        
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            # Маска шага
            mask_step = path_masks[:, i].to(device).view(batch_size, 1, 1) # bool mask
            
            # Эмбеддинги
            uncond_step = uncond_embeddings[i].expand(batch_size, -1, -1)
            s_emb = source_prompt_embeds.expand(batch_size, -1, -1)
            t_emb = target_prompt_embeds.expand(batch_size, -1, -1)
            
            # Формируем смешанный промпт
            text_input = torch.where(mask_step, t_emb, s_emb)
            
            # --- ИСПРАВЛЕНИЕ: ОБЪЕДИНЯЕМ В ОДИН ПРОГОН (Batching) ---
            # Склеиваем латенты: [latents, latents] -> размер 2B
            latents_input = torch.cat([latents] * 2)
            
            # Склеиваем промпты: [uncond, text] -> размер 2B
            prompt_embeds_input = torch.cat([uncond_step, text_input])
            
            # Один вызов UNet вместо двух
            # CUDAGraphs теперь сработает один раз и вернет большой тензор
            noise_pred = self.unet(
                latents_input, 
                t, 
                encoder_hidden_states=prompt_embeds_input
            )["sample"]
            
            # Разделяем обратно
            noise_uncond, noise_text = noise_pred.chunk(2)
            
            # Guidance
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)
            
            # Шаг шедулера
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        return latents


def prepare_shared_data(source_prompt, target_prompt, image_path, steps, device_id=0):
    device = torch.device(f"cuda:{device_id}")
    print(f"=== [Pre-computation] Running Inversion on GPU {device_id} ===")

    scheduler = DDIMScheduler(
        num_train_timesteps=1000, 
        beta_start=0.00085, beta_end=0.0120, 
        beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
    )
    
    pipeline = NullTextPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", 
        scheduler=scheduler,
        torch_dtype=torch.float32 
    ).to(device)
    
    inv_latent, uncond = pipeline.invert(image_path, source_prompt, num_inner_steps=10, num_inference_steps=steps)
    
    with torch.no_grad():
        s_emb = pipeline.encode_prompt(source_prompt, device, 1, False)[0]
        t_emb = pipeline.encode_prompt(target_prompt, device, 1, False)[0]

    print("=== [Pre-computation] Done. Moving data to CPU for sharing. ===")
    
    return {
        "inv_latent": inv_latent.detach().cpu(),
        "uncond": uncond.detach().cpu(),
        "s_emb": s_emb.detach().cpu(),
        "t_emb": t_emb.detach().cpu()
    }

# ==========================================
# 3. WORKER (Без сохранения trajectory logs)
# ==========================================

def worker_fn(rank, gpu_ids, shared_data, masks_chunk, global_indices_chunk, steps, progress_queue, output_dir):
    try:
        gpu_id = gpu_ids[rank]
        device = torch.device(f"cuda:{gpu_id}")

        os.makedirs(output_dir, exist_ok=True)

        print(f"[GPU {gpu_id}] Initializing pipeline (FP16)...")
        
        scheduler = DDIMScheduler(
            num_train_timesteps=1000, 
            beta_start=0.00085, beta_end=0.0120, 
            beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False
        )
        
        pipeline = NullTextPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", 
            scheduler=scheduler,
            torch_dtype=torch.float16
        ).to(device)
        pipeline.set_progress_bar_config(disable=True)
        # channels_last for better tensor core utilization on A100
        pipeline.unet = pipeline.unet.to(memory_format=torch.channels_last)
        pipeline.unet = torch.compile(pipeline.unet, mode="max-autotune", fullgraph=False)

        dtype = torch.float16
        inv_latent = shared_data["inv_latent"].to(device, dtype=dtype)
        uncond = shared_data["uncond"].to(device, dtype=dtype)
        s_emb = shared_data["s_emb"].to(device, dtype=dtype)
        t_emb = shared_data["t_emb"].to(device, dtype=dtype)

        BATCH_SIZE = 8  # throughput plateaus at BS=4-8; larger wastes memory with no speed gain
        num_samples = len(masks_chunk)

        def _save_image(pil_img, fpath):
            pil_img.save(fpath, quality=95)

        save_executor = ThreadPoolExecutor(max_workers=8)
        save_futures = []

        import time
        t_unet_total = 0.0
        t_vae_total = 0.0
        t_save_total = 0.0
        n_batches = 0
        print(f"[GPU {gpu_id}] Processing {num_samples} paths...")

        for i in range(0, num_samples, BATCH_SIZE):
            # Wait for previous batch's saves before reusing CPU memory
            t0_save_wait = time.perf_counter()
            for fut in save_futures:
                fut.result()
            save_futures = []
            t_save_total += time.perf_counter() - t0_save_wait

            batch_masks_np = masks_chunk[i : i + BATCH_SIZE]
            batch_indices = global_indices_chunk[i : i + BATCH_SIZE]
            current_bs = len(batch_masks_np)

            masks_tensor = torch.tensor(batch_masks_np, dtype=torch.bool, device=device)
            # Expand mask bits to diffusion steps: each bit repeated (steps // mask_bits) times
            # e.g. 20-bit [0,1,0,...] -> 40-step [0,0,1,1,0,0,...]
            masks_tensor = masks_tensor.repeat_interleave(steps // batch_masks_np.shape[1], dim=1)

            # --- SAMPLING (ONLY FINAL LATENTS) ---
            torch.cuda.synchronize(device)
            t0_unet = time.perf_counter()
            final_latents = pipeline.sample_final_only(
                source_prompt_embeds=s_emb,
                target_prompt_embeds=t_emb,
                uncond_embeddings=uncond,
                inverted_latent=inv_latent,
                path_masks=masks_tensor,
                num_inference_steps=steps
            )
            torch.cuda.synchronize(device)
            t_unet_total += time.perf_counter() - t0_unet

            # --- DECODING ---
            t0_vae = time.perf_counter()
            with torch.inference_mode():
                final_latents = final_latents.to(dtype=pipeline.vae.dtype) / pipeline.vae.config.scaling_factor
                decoded = pipeline.vae.decode(final_latents, return_dict=False)[0]
                images_gpu = (decoded / 2 + 0.5).clamp(0, 1)  # (B, 3, 512, 512) [0,1]

                # Move images to CPU for JPEG saving
                images_np = images_gpu.cpu().permute(0, 2, 3, 1).numpy()
                images_np = (images_np * 255).round().astype("uint8")
            torch.cuda.synchronize(device)
            t_vae_total += time.perf_counter() - t0_vae

            # Vectorized filenames
            mask_ints = mask_to_int_batch(batch_masks_np)
            fname_list = [f"path_{batch_indices[j]:05d}_b{mask_ints[j]}.jpg" for j in range(current_bs)]

            # Submit async JPEG saves
            pil_list = [Image.fromarray(images_np[k]) for k in range(current_bs)]
            for k in range(current_bs):
                out_path = os.path.join(output_dir, fname_list[k])
                save_futures.append(save_executor.submit(_save_image, pil_list[k], out_path))

            n_batches += 1
            if progress_queue is not None:
                progress_queue.put(current_bs)

        # Wait for remaining saves
        for fut in save_futures:
            fut.result()
        save_executor.shutdown(wait=True)

        # Timing report
        total_time = t_unet_total + t_vae_total + t_save_total
        print(f"[GPU {gpu_id}] Timing ({n_batches} batches, {num_samples} imgs):")
        print(f"  UNet sampling: {t_unet_total:.1f}s ({100*t_unet_total/total_time:.0f}%)")
        print(f"  VAE decode:    {t_vae_total:.1f}s ({100*t_vae_total/total_time:.0f}%)")
        print(f"  Save wait:     {t_save_total:.1f}s ({100*t_save_total/total_time:.0f}%)")
        print(f"  Throughput:    {num_samples/total_time:.2f} img/s (excl. warmup)")

        print(f"[GPU {gpu_id}] Finished.")

    except Exception as e:
        print(f"[GPU {rank}] ERROR: {e}")
        import traceback
        traceback.print_exc()

# ==========================================
# 4. ORCHESTRATOR
# ==========================================

def run_targeted_search(source_prompt, target_prompt, input_image, steps_total, mask_bits, num_paths_limit, gpu_ids, output_dir):
    repeat_factor = steps_total // mask_bits
    print(f"Generating {num_paths_limit} masks: {mask_bits} bits × {repeat_factor} repeat = {steps_total} diffusion steps")

    path_source = np.zeros((1, mask_bits), dtype=int)
    path_target = np.ones((1, mask_bits), dtype=int)

    total_possible = 1 << mask_bits  # 2^20 = 1,048,576
    num_random = min(max(0, num_paths_limit - 2), total_possible - 2)

    if num_random > 0:
        # Sample unique integers from [1, total_possible-2], excluding 0 (all-source) and total_possible-1 (all-target)
        rng = np.random.default_rng()
        chosen = rng.choice(total_possible - 2, size=num_random, replace=False) + 1
        # Convert integers to binary masks (MSB-first, matching mask_to_int)
        random_masks = ((chosen[:, None] >> np.arange(mask_bits - 1, -1, -1)) & 1).astype(int)
        masks = np.vstack([path_source, path_target, random_masks])
    else:
        masks = np.vstack([path_source, path_target])

    global_indices = np.arange(len(masks))
    print(f"Created {len(masks)} unique paths (out of {total_possible} possible).")

    # Предварительная подготовка
    print("\n--- Phase 1: Pre-computing Inversion (Single GPU) ---")
    shared_data = prepare_shared_data(source_prompt, target_prompt, input_image, steps_total, device_id=gpu_ids[0])
    
    torch.cuda.set_device(gpu_ids[0])
    torch.cuda.empty_cache()

    # Запуск процессов
    print(f"\n--- Phase 2: Distributed Sampling ({len(gpu_ids)} GPUs) ---")
    num_gpus = len(gpu_ids)
    
    chunks_masks = np.array_split(masks, num_gpus)
    chunks_indices = np.array_split(global_indices, num_gpus)
    
    ctx = mp.get_context('spawn')
    progress_queue = ctx.Queue()
    processes = []

    for rank in range(num_gpus):
        p = ctx.Process(
            target=worker_fn,
            args=(rank, gpu_ids, shared_data, chunks_masks[rank], chunks_indices[rank],
                  steps_total, progress_queue, output_dir)
        )
        p.daemon = True
        p.start()
        processes.append(p)

    total_paths = len(masks)
    processed_count = 0
    with tqdm(total=total_paths, unit="img", desc="Processing Paths") as pbar:
        while processed_count < total_paths:
            try:
                if not any(p.is_alive() for p in processes) and progress_queue.empty():
                    break
                
                num_done = progress_queue.get(timeout=2)
                processed_count += num_done
                pbar.update(num_done)
            except:
                pass

    for p in processes:
        p.join()

    print("Generation complete.")
    return

if __name__ == "__main__":
    # --- Настройки ---
    s_prompt = "tabby kitten walking confidently across a stone pavement."
    t_prompt = "tabby dog walking confidently across a stone pavement."
    img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "istanbul-cats-history.jpg")
    
    STEPS = 40
    MASK_BITS = 20  # search space; each bit is repeated STEPS//MASK_BITS times
    NUM_PATHS = 1_048_576  # full 2^20 paths
    OUTPUT_DIR = "/home/jovyan/shares/SR006.nfs3/svgrozny/generated_samples_40step"

    GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

    if not torch.cuda.is_available():
        print("CUDA not available. Exiting.")
        exit()
    avail_gpus = list(range(torch.cuda.device_count()))
    use_gpus = [g for g in GPUS if g in avail_gpus]
    if not use_gpus: use_gpus = [0]
    print(f"Running on GPUs: {use_gpus}")

    run_targeted_search(
        s_prompt, t_prompt, img_path,
        steps_total=STEPS,
        mask_bits=MASK_BITS,
        num_paths_limit=NUM_PATHS,
        gpu_ids=use_gpus,
        output_dir=OUTPUT_DIR
    )
    
    print("\nAll Done! Images and features saved.")