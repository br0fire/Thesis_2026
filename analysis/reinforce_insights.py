"""Deep-dive analysis of REINFORCE training dynamics.

Loads per-experiment CSVs and extracts insights about:
  - Convergence speed (when entropy crosses thresholds, when best reward appears)
  - Reward improvement trajectory (initial, midpoint, final)
  - Bit-position bias (do early bits tend to stay source across experiments?)
  - Exploration → exploitation transition
  - Reward component balance (bg_ssim vs fg_clip contribution over time)
"""
import os
import numpy as np
import pandas as pd
import torch

NFS3 = "/home/jovyan/shares/SR006.nfs3/svgrozny"

EXPERIMENTS = [
    # v1: CLIP + delta
    "test_catdog", "car_taxi", "sunflower_lavender", "chair_throne",
    "penguin_flamingo", "cake_books", "lighthouse_castle", "violin_guitar",
    "horse", "room", "snow_volcano", "butterfly_hummingbird", "sail_pirate",
    # v2clip: CLIP + relative
    "catdog_v2clip", "car_taxi_v2clip", "sunflower_lavender_v2clip", "chair_throne_v2clip",
    "violin_guitar_v2clip",
    # v2: SigLIP2 + relative
    "catdog_v2", "car_taxi_v2", "sunflower_lavender_v2", "chair_throne_v2",
    "penguin_flamingo_v2", "cake_books_v2", "lighthouse_castle_v2", "violin_guitar_v2",
    "horse_v2", "room_v2", "snow_volcano_v2", "butterfly_hummingbird_v2", "sail_pirate_v2",
]


def load_all():
    out = {}
    for name in EXPERIMENTS:
        exp_dir = os.path.join(NFS3, f"reinforce_{name}")
        csv_path = os.path.join(exp_dir, "reinforce_log.csv")
        ckpt_path = os.path.join(exp_dir, "reinforce_result.pt")
        if not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0:
            continue
        df = pd.read_csv(csv_path)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False) if os.path.isfile(ckpt_path) else None
        out[name] = {"df": df, "ckpt": ckpt}
    return out


def convergence_stats(data):
    print("=" * 88)
    print("CONVERGENCE SPEED")
    print("=" * 88)
    print(f"{'name':<20s} {'ep':>6s} {'R_init':>7s} {'R_mid':>7s} {'R_final':>7s} "
          f"{'H_50%':>7s} {'best@ep':>9s} {'best_R':>8s}")
    for name, d in data.items():
        df = d["df"]
        n = len(df)
        R = df["mean_reward"].values
        H = df["entropy"].values
        best = df["best_reward_ever"].values
        # When did the final best appear?
        final_best = best[-1]
        best_first_ep = int(np.argmax(best >= final_best - 1e-6))
        # Entropy 50% decay
        h_range = H[0] - H[-1]
        h_half = H[0] - 0.5 * h_range
        h_half_ep = int(np.argmin(np.abs(H - h_half)))
        print(f"{name:<20s} {n:>6d} {R[0]:>7.4f} {R[n//2]:>7.4f} {R[-1]:>7.4f} "
              f"{h_half_ep:>7d} {best_first_ep:>9d} {final_best:>8.4f}")


def bit_position_analysis(data):
    print()
    print("=" * 88)
    print("BIT-POSITION AVERAGES (final learned probability across experiments)")
    print("=" * 88)
    # Build matrix: experiments × 14 bits
    names = list(data.keys())
    n_bits = 14
    probs = np.zeros((len(names), n_bits))
    fg_signs = []
    for i, name in enumerate(names):
        ckpt = data[name]["ckpt"]
        p = ckpt["probs"].numpy() if torch.is_tensor(ckpt["probs"]) else ckpt["probs"]
        probs[i] = p
        # Was this a "collapsed" experiment (final fg_clip was negative)?
        final_fg = data[name]["df"]["mean_fg_clip"].values[-1]
        fg_signs.append(final_fg)

    # Split successful (fg > 0) vs collapsed
    success = np.array([fg > 0 for fg in fg_signs])
    print(f"\nSuccessful experiments (final fg_clip > 0): {success.sum()}/{len(names)}")
    print(f"  {[n for n, s in zip(names, success) if s]}")
    print(f"Collapsed experiments (final fg_clip < 0): {(~success).sum()}/{len(names)}")
    print(f"  {[n for n, s in zip(names, success) if not s]}")

    print(f"\n{'bit':<4s} {'avg_all':>9s} {'avg_success':>13s} {'avg_collapsed':>15s}")
    for b in range(n_bits):
        avg_all = probs[:, b].mean()
        avg_success = probs[success, b].mean() if success.any() else np.nan
        avg_collapsed = probs[~success, b].mean() if (~success).any() else np.nan
        marker = ""
        if avg_success > 0.7:
            marker = "  ← strongly TARGET (successful)"
        elif avg_success < 0.3:
            marker = "  ← strongly SOURCE (successful)"
        print(f"b{b:<3d} {avg_all:>9.3f} {avg_success:>13.3f} {avg_collapsed:>15.3f}{marker}")


def reward_decomposition(data):
    print()
    print("=" * 88)
    print("REWARD DECOMPOSITION (final values, α*bg + (1-α)*max(0,fg))")
    print("=" * 88)
    alpha = 0.5
    print(f"{'name':<20s} {'bg_ssim':>8s} {'fg_clip':>8s} "
          f"{'α·bg':>8s} {'(1-α)·max(0,fg)':>17s} {'total':>8s}")
    for name, d in data.items():
        df = d["df"]
        bg = df["mean_bg_ssim"].iloc[-10:].mean()
        fg = df["mean_fg_clip"].iloc[-10:].mean()
        fg_clamped = max(0, fg)
        contrib_bg = alpha * bg
        contrib_fg = (1 - alpha) * fg_clamped
        total = contrib_bg + contrib_fg
        print(f"{name:<20s} {bg:>8.3f} {fg:>+8.3f} {contrib_bg:>8.4f} {contrib_fg:>17.4f} {total:>8.4f}")


def best_vs_greedy(data):
    print()
    print("=" * 88)
    print("BEST-EVER MASK vs GREEDY FROM LEARNED POLICY")
    print("=" * 88)
    print(f"{'name':<20s} {'best_mask':<18s} {'greedy_mask':<18s} {'match':>6s}")
    for name, d in data.items():
        ckpt = d["ckpt"]
        probs = ckpt["probs"].numpy() if torch.is_tensor(ckpt["probs"]) else ckpt["probs"]
        greedy = (probs > 0.5).astype(int)
        greedy_str = "".join(str(b) for b in greedy)
        bm = ckpt.get("best_mask")
        if bm is None:
            continue
        if torch.is_tensor(bm):
            bm = bm.numpy()
        best_str = "".join(str(int(b)) for b in bm)
        match = "YES" if best_str == greedy_str else "no"
        diffs = sum(1 for a, b in zip(best_str, greedy_str) if a != b)
        print(f"{name:<20s} {best_str:<18s} {greedy_str:<18s} {match:>6s} ({diffs} bits differ)")


def exploration_vs_exploitation(data):
    print()
    print("=" * 88)
    print("EXPLORATION → EXPLOITATION TRANSITION")
    print("=" * 88)
    print(f"{'name':<20s} {'R std first 20%':>18s} {'R std last 20%':>17s} {'ratio':>8s}")
    for name, d in data.items():
        df = d["df"]
        n = len(df)
        first = df["mean_reward"].iloc[:n // 5].std()
        last = df["mean_reward"].iloc[-n // 5:].std()
        ratio = first / (last + 1e-8)
        print(f"{name:<20s} {first:>18.4f} {last:>17.4f} {ratio:>8.1f}x")


def images_vs_reward(data):
    print()
    print("=" * 88)
    print("SAMPLE EFFICIENCY (reward per image evaluated)")
    print("=" * 88)
    print(f"{'name':<20s} {'images':>8s} {'best_R':>8s} {'ep to best':>12s} "
          f"{'images to best':>16s} {'vs 16384 exh.':>14s}")
    for name, d in data.items():
        df = d["df"]
        ckpt = d["ckpt"]
        n_ep = len(df)
        total_imgs = ckpt.get("total_images", n_ep * 8)
        batch = total_imgs // n_ep if n_ep else 8
        best_reward = ckpt.get("best_reward", np.nan)
        # Episode where best was first achieved
        best_ep = int(np.argmax(df["best_reward_ever"].values >= best_reward - 1e-6))
        imgs_to_best = best_ep * batch + batch
        reduction = 16384 / imgs_to_best if imgs_to_best > 0 else float('inf')
        print(f"{name:<20s} {total_imgs:>8d} {best_reward:>8.4f} {best_ep:>12d} "
              f"{imgs_to_best:>16d} {reduction:>12.1f}x")


def main():
    data = load_all()
    if not data:
        print("No experiments loaded")
        return
    print(f"\nLoaded {len(data)} experiments\n")

    convergence_stats(data)
    bit_position_analysis(data)
    reward_decomposition(data)
    best_vs_greedy(data)
    exploration_vs_exploitation(data)
    images_vs_reward(data)


if __name__ == "__main__":
    main()
