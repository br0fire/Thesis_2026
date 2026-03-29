"""Quick test run: horse image, 10 paths, GPUs 4-7."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from test import run_targeted_search

if __name__ == "__main__":
    s_prompt = "a horse on the grass"
    t_prompt = "a robot horse on the grass"
    img_path = os.path.join(os.path.dirname(__file__), "horse.jpg")

    STEPS = 40
    MASK_BITS = 20
    NUM_PATHS = 10
    OUTPUT_DIR = "/home/jovyan/shares/SR006.nfs3/svgrozny/generated_horse_test"
    GPUS = [4, 5, 6, 7]

    if not torch.cuda.is_available():
        print("CUDA not available.")
        exit()

    avail_gpus = list(range(torch.cuda.device_count()))
    use_gpus = [g for g in GPUS if g in avail_gpus]
    if not use_gpus:
        use_gpus = [4]
    print(f"Running on GPUs: {use_gpus}")

    run_targeted_search(
        s_prompt, t_prompt, img_path,
        steps_total=STEPS,
        mask_bits=MASK_BITS,
        num_paths_limit=NUM_PATHS,
        gpu_ids=use_gpus,
        output_dir=OUTPUT_DIR,
    )
    print("\nDone! Check:", OUTPUT_DIR)
