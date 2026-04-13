"""
Synthetic DIP X-Ray Image Generator with Discriminator Quality Filter
======================================================================
Generates DIP joint images conditioned on KL score only.
Matches the new train_dip.py architecture:
  - Generator forward: (z, kl_label)       — no joint_type
  - Discriminator forward: (img, kl_label)  — no joint_type

Edit GENERATION_PLAN to control how many images per KL score.

Usage:
    python generate.py
    python generate.py --checkpoint-g cgan_output_dip/checkpoints/G_epoch_0100.pth
    python generate.py --threshold 0.01
    python generate.py --out my_folder --threshold 0.05 --max-attempts 20
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ── Import model definitions from training script ─────────────────────────────
from train_dip import Generator, Discriminator, LATENT_DIM, DEVICE

# =============================================================================
#  EDIT THIS PLAN TO CONTROL WHAT GETS GENERATED
#
#  Format:
#      KL_SCORE: NUMBER_OF_IMAGES
#
#  Valid KL scores: 0, 1, 2, 3, 4
#  Recommendation: only generate for KL=3 and KL=4 since those are the
#  rare classes you need to augment for the classifier.
# =============================================================================
GENERATION_PLAN = {
    3: 500,    # generate 200 synthetic KL=3 DIP images
    4: 500,    # generate 200 synthetic KL=4 DIP images
}


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic DIP X-ray images from trained cGAN"
    )
    parser.add_argument(
        "--checkpoint-g",
        type=str,
        default="cgan_output_dip/checkpoints/G_epoch_0100.pth",
        help="Path to Generator checkpoint (.pth file)",
    )
    parser.add_argument(
        "--checkpoint-d",
        type=str,
        default="cgan_output_dip/checkpoints/D_epoch_0100.pth",
        help="Path to Discriminator checkpoint (.pth file)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="generated_images",
        help="Output folder for generated images",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.0,
        help=(
            "Discriminator score threshold. Images scoring >= threshold are saved. "
            "Since D output is a raw WGAN score (not probability), start very low "
            "e.g. 0.0 means save everything, -10.0 saves only images D prefers."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help="Max generation attempts multiplier before giving up.",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Skip discriminator filtering entirely and save all generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_models(g_path: Path, d_path: Path, use_filter: bool):
    """Load Generator and optionally Discriminator."""
    if not g_path.exists():
        available = sorted(Path("cgan_output_dip/checkpoints").glob("G_epoch_*.pth"))
        hint = "\n".join(f"  {p.name}" for p in available) or "  (none found)"
        raise FileNotFoundError(
            f"\nGenerator checkpoint not found: {g_path}\n"
            f"Available checkpoints:\n{hint}"
        )

    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(g_path, map_location=DEVICE))
    G.eval()
    print(f"Loaded Generator      : {g_path}")

    D = None
    if use_filter:
        if not d_path.exists():
            raise FileNotFoundError(
                f"\nDiscriminator checkpoint not found: {d_path}\n"
                f"Use --no-filter to skip discriminator filtering."
            )
        D = Discriminator().to(DEVICE)
        D.load_state_dict(torch.load(d_path, map_location=DEVICE))
        D.eval()
        print(f"Loaded Discriminator  : {d_path}")

    print(f"Running on            : {DEVICE}")
    return G, D


@torch.no_grad()
def generate_and_filter(
    G: Generator,
    D,                  # can be None if no_filter=True
    kl_score: int,
    n_target: int,
    threshold: float,
    max_attempts: int,
    use_filter: bool,
    batch_size: int = 64,
):
    """
    Generate images for a single KL score, optionally filtering with D.

    Generator forward: G(z, kl_label)        — no joint_type
    Discriminator forward: D(img, kl_label)   — no joint_type

    Returns:
        saved_images : list of PIL Images
        stats        : generation statistics dict
    """
    if kl_score not in range(5):
        raise ValueError(f"KL score must be 0-4, got {kl_score}")

    saved_images    = []
    total_generated = 0
    total_passed    = 0
    total_failed    = 0
    max_total_gen   = n_target * max_attempts

    while len(saved_images) < n_target and total_generated < max_total_gen:
        bs   = min(batch_size, n_target - len(saved_images),
                   max_total_gen - total_generated)
        z    = torch.randn(bs, LATENT_DIM, device=DEVICE)
        kl_t = torch.full((bs,), kl_score, dtype=torch.long, device=DEVICE)

        # ── Generate ──────────────────────────────────────────────────────────
        fake_imgs = G(z, kl_t)                            # (bs, 1, 180, 180)

        # ── Filter with Discriminator (optional) ──────────────────────────────
        if use_filter and D is not None:
            d_scores    = D(fake_imgs, kl_t).squeeze(1)  # (bs,) raw WGAN scores
            passed_mask = d_scores >= threshold
        else:
            passed_mask = torch.ones(bs, dtype=torch.bool, device=DEVICE)

        total_generated += bs
        total_passed    += passed_mask.sum().item()
        total_failed    += (~passed_mask).sum().item()

        # ── Convert passing images to PIL ─────────────────────────────────────
        good_imgs = fake_imgs[passed_mask].cpu()
        good_imgs = (good_imgs * 0.5 + 0.5).clamp(0, 1)  # [-1,1] -> [0,1]
        good_imgs = (good_imgs * 255).byte().squeeze(1)   # -> (n, 180, 180) uint8

        for i in range(good_imgs.size(0)):
            if len(saved_images) < n_target:
                saved_images.append(
                    Image.fromarray(good_imgs[i].numpy(), mode="L")
                )

    stats = {
        "target"          : n_target,
        "saved"           : len(saved_images),
        "total_generated" : total_generated,
        "pass_rate"       : total_passed / total_generated if total_generated > 0 else 0,
        "hit_limit"       : total_generated >= max_total_gen and len(saved_images) < n_target,
    }
    return saved_images, stats


def print_plan(plan: dict, out_dir: Path, threshold: float,
               max_attempts: int, use_filter: bool):
    print("\nGeneration plan (DIP joints only):")
    print(f"  {'KL Score':<10}  {'Target':>7}")
    print(f"  {'-'*10}  {'-'*7}")
    total = 0
    for kl, n in plan.items():
        print(f"  KL={kl:<7}  {n:>7} images")
        total += n
    print(f"  {'':14}  -------")
    print(f"  {'Total':<14}  {total:>7} images")
    if use_filter:
        print(f"\n  Filter          : D score >= {threshold}  (WGAN raw score, not probability)")
        print(f"  Max attempts    : {max_attempts}x per target")
    else:
        print(f"\n  Filter          : disabled (saving all generated images)")
    print(f"  Output folder   : {out_dir}/\n")
    return total


def run(plan, g_path, d_path, out_dir, threshold, max_attempts, use_filter, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    G, D  = load_models(g_path, d_path, use_filter)
    total = print_plan(plan, out_dir, threshold, max_attempts, use_filter)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved_so_far = 0
    all_stats    = []

    for kl, n_target in plan.items():
        images, stats = generate_and_filter(
            G, D,
            kl_score     = kl,
            n_target     = n_target,
            threshold    = threshold,
            max_attempts = max_attempts,
            use_filter   = use_filter,
        )

        # Save images — filename includes KL score for easy identification
        for i, img in enumerate(images, start=1):
            filename = out_dir / f"DIP_KL{kl}_{i:04d}.png"
            img.save(filename)

        saved_so_far += stats["saved"]
        all_stats.append((kl, stats))

        warn = "  !! INCOMPLETE" if stats["hit_limit"] else ""
        rate = f"(pass rate: {stats['pass_rate']*100:.1f}%)" if use_filter else ""
        print(
            f"  [{saved_so_far:>4}/{total}]  "
            f"KL={kl}  ->  "
            f"{stats['saved']}/{n_target} saved  |  "
            f"generated {stats['total_generated']}  "
            f"{rate}{warn}"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_gen  = sum(s["total_generated"] for _, s in all_stats)
    total_save = sum(s["saved"]           for _, s in all_stats)

    print(f"\n{'='*55}")
    print(f"  Total generated : {total_gen}")
    print(f"  Total saved     : {total_save}")
    if use_filter:
        overall = total_save / total_gen * 100 if total_gen > 0 else 0
        print(f"  Overall pass rate: {overall:.1f}%")
        print(f"  Total discarded : {total_gen - total_save}")

    incomplete = [(kl, s) for kl, s in all_stats if s["hit_limit"]]
    if incomplete:
        print(f"\n  WARNING: {len(incomplete)} class(es) did not reach target.")
        print(f"  Try lowering --threshold or raising --max-attempts:")
        for kl, s in incomplete:
            print(f"    KL={kl}: only saved {s['saved']}/{s['target']}")
    else:
        print(f"\n  All targets met!")
    print(f"  Output folder   : {out_dir}/")
    print(f"{'='*55}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    run(
        plan         = GENERATION_PLAN,
        g_path       = Path(args.checkpoint_g),
        d_path       = Path(args.checkpoint_d),
        out_dir      = Path(args.out),
        threshold    = args.threshold,
        max_attempts = args.max_attempts,
        use_filter   = not args.no_filter,
        seed         = args.seed,
    )