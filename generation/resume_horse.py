"""
Resume horse-robot generation: produce only MISSING b-values.

Safety:
  - Scans existing files, generates only what's missing
  - Features saved to feature_dictionary_horse_resume.pkl (no overwrite)
  - Graceful stop: Ctrl+C or `kill <PID>` — already-saved images are safe
  - Creates a PID file for easy stopping: kill $(cat resume_horse.pid)

Usage:
  nohup python resume_horse.py > resume_horse.log 2>&1 &
  kill $(cat resume_horse.pid)   # to stop gracefully
"""
import sys, os
import signal
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import torch
from test import run_targeted_search, mask_to_int_batch

OUTPUT_DIR = "/home/jovyan/shares/SR006.nfs3/svgrozny/generated_horse_300k"
FEATURE_OUTPUT = os.path.join(PROJECT_DIR, "analysis", "feature_dictionary_horse_resume.pkl")
PID_FILE = os.path.join(PROJECT_DIR, "resume_horse.pid")

S_PROMPT = "a horse on the grass"
T_PROMPT = "a robot horse on the grass"
IMG_PATH = os.path.join(SCRIPT_DIR, "horse.jpg")

STEPS = 40
MASK_BITS = 20
TOTAL_POSSIBLE = 1 << MASK_BITS  # 1,048,576
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]


def get_existing_b_values(images_dir):
    """Scan output dir for existing b-values."""
    existing = set()
    for f in os.listdir(images_dir):
        if f.endswith(".jpg"):
            try:
                b = int(f.rsplit("_b", 1)[1].split(".")[0])
                existing.add(b)
            except (IndexError, ValueError):
                pass
    return existing


def b_to_mask(b, mask_bits=20):
    """Convert integer b to binary mask (MSB-first, matching test.py)."""
    return ((b >> np.arange(mask_bits - 1, -1, -1)) & 1).astype(int)


def main():
    # Write PID file for easy stopping
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))
    print(f"PID: {os.getpid()} (stop with: kill $(cat {PID_FILE}))")

    # Graceful shutdown
    def _cleanup(sig, frame):
        print(f"\nReceived signal {sig}, shutting down gracefully...")
        if os.path.exists(PID_FILE):
            os.remove(PID_FILE)
        sys.exit(0)
    signal.signal(signal.SIGTERM, _cleanup)
    signal.signal(signal.SIGINT, _cleanup)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find missing b-values
    print("Scanning existing images...")
    existing = get_existing_b_values(OUTPUT_DIR)
    print(f"Existing: {len(existing)} / {TOTAL_POSSIBLE}")

    all_b = set(range(TOTAL_POSSIBLE))
    missing_b = sorted(all_b - existing)
    print(f"Missing: {len(missing_b)}")

    if not missing_b:
        print("All paths already generated!")
        os.remove(PID_FILE)
        return

    # Convert missing b-values to masks
    masks = np.array([b_to_mask(b, MASK_BITS) for b in missing_b])
    global_indices = np.arange(len(masks))

    # Verify roundtrip: mask → int should give back original b
    roundtrip = mask_to_int_batch(masks)
    assert np.array_equal(roundtrip, np.array(missing_b)), "Mask encoding roundtrip failed!"
    print(f"Roundtrip check passed. Generating {len(missing_b)} paths...")

    # Prepare shared data (inversion on first GPU)
    from test import prepare_shared_data, worker_fn
    import torch.multiprocessing as mp
    import pickle
    from tqdm import tqdm

    avail_gpus = list(range(torch.cuda.device_count()))
    use_gpus = [g for g in GPUS if g in avail_gpus]
    if not use_gpus:
        use_gpus = [0]

    print(f"\n--- Phase 1: Pre-computing Inversion (GPU {use_gpus[0]}) ---")
    shared_data = prepare_shared_data(S_PROMPT, T_PROMPT, IMG_PATH, STEPS, device_id=use_gpus[0])

    torch.cuda.set_device(use_gpus[0])
    torch.cuda.empty_cache()

    print("Pre-downloading DINOv2...")
    torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', force_reload=False)

    print(f"\n--- Phase 2: Distributed Sampling ({len(use_gpus)} GPUs) ---")
    num_gpus = len(use_gpus)
    chunks_masks = np.array_split(masks, num_gpus)
    chunks_indices = np.array_split(global_indices, num_gpus)

    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    progress_queue = ctx.Queue()
    processes = []

    for rank in range(num_gpus):
        p = ctx.Process(
            target=worker_fn,
            args=(rank, use_gpus, shared_data, chunks_masks[rank], chunks_indices[rank],
                  STEPS, queue, progress_queue, OUTPUT_DIR)
        )
        p.daemon = True
        p.start()
        processes.append(p)

    total_paths = len(masks)
    processed_count = 0
    with tqdm(total=total_paths, unit="img", desc="Resume Horse") as pbar:
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

    # Merge features → separate file (no overwrite)
    print("Merging feature logs...")
    collected_files = []
    while not queue.empty():
        collected_files.append(queue.get())

    pkl_files = [p for p in collected_files if p.endswith(".pkl")]
    feature_dict = {}
    for fp in pkl_files:
        try:
            with open(fp, "rb") as f:
                d = pickle.load(f)
            feature_dict.update(d)
            os.remove(fp)
        except Exception as e:
            print(f"Error loading {fp}: {e}")

    with open(FEATURE_OUTPUT, "wb") as f:
        pickle.dump(feature_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Resume features saved to {FEATURE_OUTPUT} ({len(feature_dict)} entries)")

    # Cleanup
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)

    total_now = len(get_existing_b_values(OUTPUT_DIR))
    print(f"\nDone! Total images now: {total_now} / {TOTAL_POSSIBLE}")


if __name__ == "__main__":
    main()
