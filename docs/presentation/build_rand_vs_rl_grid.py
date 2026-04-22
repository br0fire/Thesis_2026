"""
Build a 'random vs REINFORCE at small budget' visual grid.

For 3 of the 8 canonical edits, at budget = 80 generations (episode 10):
  - random-80   : best of 80 uniform random masks (from exhaustive lookup).
  - REINFORCE-80: best-ever mask reached by REINFORCE by episode 10
                  (found by matching the CSV's best_reward_ever to the exhaustive table).

Saves the 3x3 grid (source | random@80 | REINFORCE@80) with rewards as titles.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RF_ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/reinforce_a05"
EX_ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/exhaustive"
IM_ROOT = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/exhaustive_images"
CANON   = "/home/jovyan/shares/SR006.nfs3/svgrozny/clean_v1/canonical"

ALPHA = 0.5
BUDGET = 80           # masks seen
EPISODES_BUDGET = BUDGET // 8  # CSV "episode" index corresponding to budget

OUT = "/home/jovyan/shares/SR006.nfs2/svgrozny/project/clear_project/docs/presentation/figures/rand_vs_rl_budget80.png"


def reward_of(bg, fg, alpha=ALPHA):
    sig = 1.0 / (1.0 + np.exp(-fg * 10.0))
    return np.maximum(bg, 1e-6) ** alpha * np.maximum(sig, 1e-6) ** (1 - alpha)


def load_source(exp):
    # canonical source is a torch tensor; we just want a uint8 image for plotting
    import torch
    src = torch.load(f"{CANON}/{exp}/source.pt", map_location="cpu", weights_only=False)
    if isinstance(src, dict):
        src = src.get("image", src.get("source", list(src.values())[0]))
    if hasattr(src, "numpy"):
        src = src.numpy()
    arr = np.asarray(src)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[0] == 3:
        arr = arr.transpose(1, 2, 0)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    return arr


def best_random_id(rewards_all, budget, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, len(rewards_all), size=budget)
    best_pos = ids[np.argmax(rewards_all[ids])]
    return int(best_pos), float(rewards_all[best_pos])


def find_rl_best_id(rewards_all, target_reward, tol=1e-4):
    """Return mask id whose reward matches target (REINFORCE's best_reward_ever)."""
    diffs = np.abs(rewards_all - target_reward)
    pos = int(diffs.argmin())
    return pos, float(rewards_all[pos])


def best_mask_id(rewards_all):
    pos = int(np.argmax(rewards_all))
    return pos, float(rewards_all[pos])


def main():
    # At budget 80: REINFORCE is at ~99.9-100% of ceiling; random is visibly behind.
    experiments = [
        ("bgrich_teapot_samovar",        0),   # Δ ≈ +0.111, RL = 100% of ceiling
        ("bgrich_guitar_banjo",         44),   # Δ ≈ +0.083, RL = 99.9%
        ("bgrich_telescope_microscope", 44),   # Δ ≈ +0.064, RL = 99.4%
    ]

    NCOL = 5
    fig, axes = plt.subplots(len(experiments), NCOL, figsize=(17, 3.6 * len(experiments)))

    all_ones_id = (1 << 14) - 1   # mask = 1111…1, 14 bits set

    for i, (exp, rand_seed) in enumerate(experiments):
        bg = np.load(f"{EX_ROOT}/{exp}/bg_ssim.npy")
        fg = np.load(f"{EX_ROOT}/{exp}/fg_clip.npy")
        R = reward_of(bg, fg)

        src_im = load_source(exp)

        all_ones_r = float(R[all_ones_id])
        rand_id, rand_r = best_random_id(R, BUDGET, seed=rand_seed)

        df = pd.read_csv(f"{RF_ROOT}/{exp}/reinforce_log.csv")
        ep = min(EPISODES_BUDGET - 1, len(df) - 1)
        tgt = float(df["best_reward_ever"].iloc[ep])
        rl_id, rl_r = find_rl_best_id(R, tgt)

        ceil_id, ceil_r = best_mask_id(R)

        mm = np.load(f"{IM_ROOT}/{exp}/all_images.npy", mmap_mode="r")
        ones_im = np.asarray(mm[all_ones_id]).transpose(1, 2, 0)
        rand_im = np.asarray(mm[rand_id]).transpose(1, 2, 0)
        rl_im   = np.asarray(mm[rl_id]).transpose(1, 2, 0)
        ceil_im = np.asarray(mm[ceil_id]).transpose(1, 2, 0)

        pretty_map = {
            "bgrich_teapot_samovar":       "teapot → samovar",
            "bgrich_guitar_banjo":         "guitar → banjo",
            "bgrich_telescope_microscope": "telescope → microscope",
        }
        pretty = pretty_map.get(exp, exp.replace("bgrich_", "").replace("_", " → "))

        axes[i, 0].imshow(src_im);  axes[i, 0].set_title(f"source • {pretty}", fontsize=13)
        axes[i, 1].imshow(ones_im); axes[i, 1].set_title(f"all-target baseline\nR = {all_ones_r:.3f}", fontsize=13, color="#A23B3B")
        axes[i, 2].imshow(rand_im); axes[i, 2].set_title(f"random (budget {BUDGET})\nR = {rand_r:.3f}", fontsize=13)
        axes[i, 3].imshow(rl_im);   axes[i, 3].set_title(f"REINFORCE (budget {BUDGET})\nR = {rl_r:.3f}", fontsize=13, color="#1F4E79")
        axes[i, 4].imshow(ceil_im); axes[i, 4].set_title(f"exhaustive max ($2^{{14}}$)\nR = {ceil_r:.3f}", fontsize=13, color="#666")
        for j in range(NCOL):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])

    fig.tight_layout()
    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print("saved:", OUT)


if __name__ == "__main__":
    main()
