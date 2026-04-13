# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project exploring **binary diffusion path search** — generating images by interpolating between source and target prompts at each diffusion step using binary masks. Each bit in an N-bit mask selects whether a diffusion step uses the source or target prompt. The project exhaustively generates images for all (or sampled) binary paths, then evaluates and clusters them.

Two generation backends:
- **SD 1.5** (`generation/test.py`): Uses null-text inversion from a source image + DDIM. Requires an input image.
- **FLUX.2-klein-base-9B** (`generation/flux_generate.py`): Text-only, no inversion. Uses CFG with empty negative prompt. Shared random noise across all paths.

## Directory Structure

```
generation/          # Image generation scripts
  flux_generate.py   #   FLUX.2 binary path generator
  test.py            #   SD 1.5 null-text inversion generator
  data/              #   Source images + cached inversions (.pt)
metrics/             # Evaluation scripts
  segment_source.py  #   Foreground segmentation (CLIPSeg / SAM)
  calc_seg_metrics.py#   Segmentation-based metrics (multi-GPU)
  calc_bg_metrics.py #   Background metrics (single-GPU)
  calc_metrics.py    #   CLIP score metrics (single-GPU)
  extract_dino_features.py  # DINOv2 feature extraction
  masks/             #   Output: background masks (.npy + _vis.jpg)
  results/           #   Output: metric CSVs
analysis/            # Visualization and analysis
  visualize_clusters.py  # UMAP + HDBSCAN clustering
  analyze_bits.py        # Bit-position correlation analysis
  make_presentation.py   # Presentation generator
  embeddings/            # Output: DINOv2 feature dictionaries (.pkl)
  *_analysis/            # Per-experiment output dirs (coords, labels, grids)
scripts/             # Pipeline runner scripts
  run_flux_pipeline.sh       # Universal FLUX pipeline (use this)
  run_flux_10_experiments.sh # Batch runner for 10 experiments
  monitor_and_continue.sh    # Pipeline monitoring utility
  legacy/                    # Old case-specific pipeline scripts
logs/                # Pipeline run logs
trash/               # Deprecated scripts and data
```

## Architecture

The pipeline has 5 sequential stages, orchestrated by `scripts/run_flux_pipeline.sh`:

1. **Generation** — Multi-GPU (8×A100), batched diffusion with `torch.multiprocessing.spawn`. Each worker loads the model independently. Text embeddings pre-computed once on GPU 0, then shared via CPU tensors.

2. **Segmentation** (`metrics/segment_source.py`) — Produces `background_mask.npy` (1=bg, 0=fg). Two methods: CLIPSeg-only (fast/coarse) or CLIPSeg+SAM union (precise).

3. **Seg metrics** (`metrics/calc_seg_metrics.py`) — Multi-GPU. Computes per-image: `bg_clip_similarity`, `bg_ssim`, `fg_clip_score` (using delta text embedding: normalized target−source). Outputs CSV.

4. **DINOv2 features** (`metrics/extract_dino_features.py`) — Multi-GPU. Extracts `dinov2_vits14` embeddings → `feature_dictionary.pkl`.

5. **Analysis** (`analysis/visualize_clusters.py`) — UMAP (GPU via cuML preferred) + HDBSCAN clustering. Produces cluster maps, metric heatmaps, and image grids. `analysis/analyze_bits.py` computes bit-position correlations with metrics.

## Environment

```bash
PYTHON=/home/jovyan/.mlspace/envs/svgrozny_base/bin/python
export LD_LIBRARY_PATH=/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/libcuml/lib64:/home/jovyan/.mlspace/envs/svgrozny_base/lib/python3.11/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export HF_HOME=/home/jovyan/shares/SR006.nfs3/svgrozny/hf_cache
```

## Running the Pipeline

The universal pipeline script is `scripts/run_flux_pipeline.sh`, driven by env vars:

```bash
SOURCE_PROMPT="a photo of a cat" \
TARGET_PROMPT="a photo of a dog" \
SEG_PROMPT="cat" \
NAME="catdog" \
N_BITS=14 \
SEED=42 \
bash scripts/run_flux_pipeline.sh
```

Resume from a specific step with `START_STEP=3`. Generated images go to `/home/jovyan/shares/SR006.nfs3/svgrozny/generated_flux_${NAME}`.

Individual scripts can be run standalone — all accept `--gpus`, `--batch_size`, `--num_workers` flags. Multi-GPU scripts default to 8 GPUs (0-7).

### REINFORCE Search (alternative to exhaustive generation)

Instead of generating all 2^N_BITS paths, learn the optimal bit distribution via REINFORCE:

```bash
$PYTHON generation/reinforce_search.py \
    --source_prompt "a tabby cat walking on stone pavement, photo" \
    --target_prompt "a golden retriever dog walking on stone pavement, photo" \
    --seg_prompt "cat" \
    --output_dir /path/to/output \
    --gpu 0 --n_bits 14 --batch_size 8 --top_k 10
```

Uses ~1-2K images instead of 16K. Policy is N_BITS independent Bernoulli logit parameters optimized with REINFORCE. Reward is `alpha * bg_ssim + (1-alpha) * fg · normalize(tgt_text - src_text)`.

Defaults (tuned from a 29-experiment analysis):
- `vision_model=google/siglip2-so400m-patch14-384` (reward VLM; SigLIP 2 SO400M at 384px)
- `num_episodes=300`, `min_episodes=200` floor before any early-stop
- `alpha=0.3`, `entropy_coeff=0.05`
- `plateau_patience=150` — stop if best reward hasn't improved for 150 episodes
- `entropy_stop=0.5` — stop when policy entropy drops below 0.5
- `normalize_advantages=True` (standardize per-batch for stable gradients)

## Key Conventions

- **File naming**: Generated images are `path_XXXXX_bN.jpg` where N is the integer representation of the binary mask (MSB-first).
- **Mask convention**: Background mask arrays use 1=background, 0=foreground.
- **Combined score**: Geometric mean of min-max normalized bg_ssim and fg_clip_score: `sqrt(norm_bg * norm_fg)`.
- **N_BITS**: Configurable (default 14 for FLUX, 20 for SD). Total diffusion steps = N_BITS × repeat_factor (e.g., 14 bits × 2 = 28 steps).
- **Data storage**: Large files (images, pickles, models) live on NFS3 (`/home/jovyan/shares/SR006.nfs3/svgrozny/`), code on NFS2.
- All multi-GPU scripts use `mp.get_context('spawn')` — required for CUDA fork safety.

## Existing Experiments

Three completed experiments: `catdog` (cat→dog, 1M images, 20-bit SD), `horse` (horse→zebra, 300K), `room` (room editing, 1M). FLUX experiments use 14-bit paths with various prompts.
