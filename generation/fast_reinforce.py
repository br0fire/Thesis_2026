"""Fast REINFORCE using pre-computed bg_ssim/fg_clip arrays from exhaustive.

No FLUX / SigLIP / GPU needed at all — rewards are O(1) lookups. Entire
80-episode training runs in seconds on CPU.

Input:
  --exhaustive_dir/bg_ssim.npy  # (2^n_bits,) float32
  --exhaustive_dir/fg_clip.npy

Output:
  --output_dir/reinforce_log.csv
  --output_dir/reinforce_result.pt
  --output_dir/top_k/top_*.png   (optional — loads specific images from images.npy memmap)
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def mask_to_int_batch(masks: torch.Tensor) -> torch.Tensor:
    """masks: (B, n_bits) {0,1} float or bool → (B,) int64 (MSB-first)."""
    B, n = masks.shape
    bits = torch.arange(n - 1, -1, -1, dtype=torch.int64, device=masks.device)
    weights = (1 << bits)
    return (masks.to(torch.int64) * weights).sum(dim=1)


def int_to_mask_np(n, n_bits):
    arr = np.zeros(n_bits, dtype=np.float32)
    for i in range(n_bits):
        if n & (1 << (n_bits - 1 - i)):
            arr[i] = 1.0
    return arr


class BernoulliPolicy(torch.nn.Module):
    def __init__(self, n_bits, init_logit=0.0):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.full((n_bits,), float(init_logit)))

    def probs(self):
        return torch.sigmoid(self.logits)

    def sample(self, batch_size):
        p = self.probs()
        dist = torch.distributions.Bernoulli(p.unsqueeze(0).expand(batch_size, -1))
        m = dist.sample()
        log_p = dist.log_prob(m).sum(dim=1)
        return m, log_p

    def entropy(self):
        p = self.probs().clamp(1e-9, 1 - 1e-9)
        return -(p * p.log() + (1 - p) * (1 - p).log()).sum()


def compute_reward(bg_ssim, fg_clip, alpha):
    fg_sig = 1.0 / (1.0 + np.exp(-fg_clip * 10.0))
    return (np.clip(bg_ssim, 1e-6, None) ** alpha
            * np.clip(fg_sig, 1e-6, None) ** (1 - alpha))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exhaustive_dir", required=True,
                    help="Dir with bg_ssim.npy + fg_clip.npy")
    ap.add_argument("--output_dir",     required=True)
    ap.add_argument("--images_npy",     default=None,
                    help="Optional path to all_images.npy for top-K PNG extraction")
    ap.add_argument("--n_bits",         type=int, default=14)
    ap.add_argument("--num_episodes",   type=int, default=80)
    ap.add_argument("--batch_size",     type=int, default=8)
    ap.add_argument("--lr",             type=float, default=0.10)
    ap.add_argument("--alpha",          type=float, default=0.5)
    ap.add_argument("--entropy_coeff",  type=float, default=0.05)
    ap.add_argument("--baseline_ema",   type=float, default=0.9)
    ap.add_argument("--normalize_advantages", action="store_true", default=True)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--top_k",          type=int, default=10)
    ap.add_argument("--log_interval",   type=int, default=10)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load precomputed reward components
    bg_all = np.load(os.path.join(args.exhaustive_dir, "bg_ssim.npy"))
    fg_all = np.load(os.path.join(args.exhaustive_dir, "fg_clip.npy"))
    total = 1 << args.n_bits
    assert len(bg_all) == total, f"bg_ssim length {len(bg_all)} != 2^{args.n_bits}={total}"
    print(f"Loaded exhaustive components: {total} rewards, alpha={args.alpha}", flush=True)

    rewards_all = compute_reward(bg_all, fg_all, args.alpha).astype(np.float32)
    print(f"  Reward at α={args.alpha}: min={rewards_all.min():.4f} max={rewards_all.max():.4f} "
          f"mean={rewards_all.mean():.4f}", flush=True)

    policy = BernoulliPolicy(args.n_bits)
    opt = torch.optim.Adam([policy.logits], lr=args.lr)

    # Training log CSV
    log_path = os.path.join(args.output_dir, "reinforce_log.csv")
    fh = open(log_path, "w", newline="")
    writer = csv.writer(fh)
    header = (["episode", "mean_reward", "best_reward_ever",
               "mean_bg_ssim", "mean_fg_clip", "entropy", "baseline"]
              + [f"prob_{i}" for i in range(args.n_bits)])
    writer.writerow(header)

    baseline = 0.0
    best_reward = -1.0
    best_mask_int = -1
    mean_rewards = []

    t0 = time.perf_counter()
    for ep in range(args.num_episodes):
        masks, log_probs = policy.sample(args.batch_size)  # (B, n_bits), (B,)
        mask_ints = mask_to_int_batch(masks).cpu().numpy()
        bg_vals = bg_all[mask_ints]
        fg_vals = fg_all[mask_ints]
        r_vals = rewards_all[mask_ints]
        rewards = torch.from_numpy(r_vals).float()

        mean_r = float(rewards.mean())
        mean_rewards.append(mean_r)

        baseline = args.baseline_ema * baseline + (1 - args.baseline_ema) * mean_r
        adv = rewards - baseline
        if args.normalize_advantages and adv.numel() > 1:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        policy_loss = -(adv.detach() * log_probs).mean()
        ent = policy.entropy()
        loss = policy_loss - args.entropy_coeff * ent

        opt.zero_grad(); loss.backward(); opt.step()

        # Track best
        bi = int(np.argmax(r_vals))
        if r_vals[bi] > best_reward:
            best_reward = float(r_vals[bi])
            best_mask_int = int(mask_ints[bi])

        probs = policy.probs().detach().cpu().numpy()
        writer.writerow(
            [ep, f"{mean_r:.6f}", f"{best_reward:.6f}",
             f"{bg_vals.mean():.6f}", f"{fg_vals.mean():.6f}",
             f"{ent.item():.4f}", f"{baseline:.6f}"] + [f"{p:.4f}" for p in probs])
        fh.flush()

        if ep % args.log_interval == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [{ep:3d}/{args.num_episodes}] R={mean_r:.4f} best={best_reward:.4f} "
                  f"H={ent.item():.2f}  ({elapsed:.1f}s)", flush=True)
    fh.close()

    # Save result
    probs_final = policy.probs().detach().cpu().numpy()
    result = {
        "method": "fast_reinforce",
        "probs": torch.from_numpy(probs_final),
        "best_reward": best_reward,
        "best_mask_int": best_mask_int,
        "best_mask": torch.from_numpy(int_to_mask_np(best_mask_int, args.n_bits)),
        "total_images": args.num_episodes * args.batch_size,
        "args": vars(args),
    }
    torch.save(result, os.path.join(args.output_dir, "reinforce_result.pt"))
    print(f"\nFinal: best_reward={best_reward:.4f}  mask_int={best_mask_int}  "
          f"time={time.perf_counter() - t0:.1f}s", flush=True)

    # Optional top-K PNG extraction
    if args.images_npy and os.path.isfile(args.images_npy):
        print(f"\nSaving top-{args.top_k} PNGs from {args.images_npy}", flush=True)
        imgs = np.load(args.images_npy, mmap_mode="r")
        # Greedy mask + sample (top_k-1) extras from final policy
        greedy_int = int(mask_to_int_batch(
            (policy.probs() > 0.5).float().unsqueeze(0)).item())
        masks_to_save = [("best", best_mask_int), ("greedy", greedy_int)]
        # Sample additional from policy
        torch.manual_seed(args.seed + 1)
        with torch.no_grad():
            extra_masks, _ = policy.sample(max(args.top_k - 2, 0))
        for i, mask in enumerate(extra_masks):
            mi = int(mask_to_int_batch(mask.unsqueeze(0)).item())
            masks_to_save.append((f"sample_{i}", mi))
        topk_dir = os.path.join(args.output_dir, "top_k")
        os.makedirs(topk_dir, exist_ok=True)
        for tag, mi in masks_to_save:
            img = np.array(imgs[mi]).transpose(1, 2, 0)  # (H, W, 3)
            r = float(rewards_all[mi])
            Image.fromarray(img).save(
                os.path.join(topk_dir, f"{tag}_r{r:.4f}_b{mi}.png"))


if __name__ == "__main__":
    main()
