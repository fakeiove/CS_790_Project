"""
Synthetic X-Ray Image Generator with Discriminator Quality Filter
==================================================================
Generate any number of images for any (joint_type, KL score) combination
by editing the GENERATION_PLAN dictionary below.

Each generated image is passed through the Discriminator.
Only images that FOOL the Discriminator (score >= threshold) are saved.

Usage:
    python generate.py
    python generate.py --checkpoint-g cgan_output_0404/checkpoints/G_epoch_0030.pth
    python generate.py --threshold 0.3
    python generate.py --out my_folder --threshold 0.4 --max-attempts 20
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# ── Import model definitions from your training script ───────────────────────
from train import Generator, Discriminator, LATENT_DIM, DEVICE

# =============================================================================
#  EDIT THIS PLAN TO CONTROL WHAT GETS GENERATED
#
#  Format:
#      ("JOINT_TYPE", KL_SCORE): NUMBER_OF_IMAGES
#
#  Valid joint types : "DIP", "PIP", "MCP"
#  Valid KL scores   : 0, 1, 2, 3, 4
# =============================================================================
GENERATION_PLAN = {
    ("DIP", 3): 20,
    ("DIP", 4):  6,
    ("PIP", 3): 20,
    ("PIP", 4):  6,
    ("MCP", 3): 30,
    ("MCP", 4):  8,
}

# ── Mapping (must match what was used during training) ────────────────────────
JOINT_TYPE_MAP = {"DIP": 0, "PIP": 1, "MCP": 2}


# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate synthetic X-ray images from trained cGAN"
    )
    parser.add_argument(
        "--checkpoint-g",
        type=str,
        default="cgan_output_0404/checkpoints/G_epoch_0030.pth",
        help="Path to Generator checkpoint (.pth file)",
    )
    parser.add_argument(
        "--checkpoint-d",
        type=str,
        default="cgan_output_0404/checkpoints/D_epoch_0030.pth",
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
        default=0.2,
        help=(
            "Discriminator score threshold (0.0-1.0). "
            "Images scoring >= threshold are saved. "
            "Start low (0.3) if your D is very strong."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=10,
        help=(
            "Max generation attempts multiplier before giving up. "
            "e.g. 10 means: try generating up to 10x the target count."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def load_models(g_path: Path, d_path: Path):
    """Load both Generator and Discriminator from checkpoints."""
    for path in [g_path, d_path]:
        if not path.exists():
            available = sorted(Path("cgan_output_0404/checkpoints").glob("*.pth"))
            hint = "\n".join(f"  {p.name}" for p in available) or "  (none found)"
            raise FileNotFoundError(
                f"\nCheckpoint not found: {path}\n"
                f"Available checkpoints:\n{hint}"
            )

    G = Generator().to(DEVICE)
    G.load_state_dict(torch.load(g_path, map_location=DEVICE))
    G.eval()

    D = Discriminator().to(DEVICE)
    D.load_state_dict(torch.load(d_path, map_location=DEVICE))
    D.eval()

    print(f"Loaded Generator      : {g_path}")
    print(f"Loaded Discriminator  : {d_path}")
    print(f"Running on            : {DEVICE}")
    return G, D


@torch.no_grad()
def generate_and_filter(
    G: Generator,
    D: Discriminator,
    joint_type: str,
    kl_score: int,
    n_target: int,
    threshold: float,
    max_attempts: int,
    batch_size: int = 64,
):
    """
    Keep generating batches until we have n_target images that pass the
    Discriminator filter, or until we hit max_attempts * n_target total
    generations (safety ceiling to avoid infinite loops).

    D score >= threshold  ->  D thinks it MIGHT be real  ->  SAVE   (good quality)
    D score <  threshold  ->  D easily spots it as fake  ->  DISCARD (bad quality)

    Returns:
        saved_images : list of PIL Images that passed the filter
        stats        : dict with generation statistics
    """
    joint_upper     = joint_type.upper()
    jt_id           = JOINT_TYPE_MAP[joint_upper]
    saved_images    = []
    total_generated = 0
    total_passed    = 0
    total_failed    = 0
    max_total_gen   = n_target * max_attempts   # hard ceiling to avoid infinite loop

    while len(saved_images) < n_target and total_generated < max_total_gen:
        bs   = min(batch_size, n_target - len(saved_images), max_total_gen - total_generated)
        z    = torch.randn(bs, LATENT_DIM, device=DEVICE)
        kl_t = torch.full((bs,), kl_score, dtype=torch.long, device=DEVICE)
        jt_t = torch.full((bs,), jt_id,    dtype=torch.long, device=DEVICE)

        # ── Generate ──────────────────────────────────────────────────────────
        fake_imgs = G(z, kl_t, jt_t)                     # (bs, 1, 180, 180) [-1,1]

        # ── Score with Discriminator ──────────────────────────────────────────
        d_scores = D(fake_imgs, kl_t, jt_t).squeeze(1)   # (bs,)  range [0,1]

        # ── Filter ────────────────────────────────────────────────────────────
        passed_mask = d_scores >= threshold               # True = passes filter

        total_generated += bs
        total_passed    += passed_mask.sum().item()
        total_failed    += (~passed_mask).sum().item()

        # Convert passing images to PIL and collect
        good_imgs = fake_imgs[passed_mask].cpu()          # (n_pass, 1, 180, 180)
        good_imgs = (good_imgs * 0.5 + 0.5).clamp(0, 1)  # -> [0, 1]
        good_imgs = (good_imgs * 255).byte().squeeze(1)   # -> (n_pass, 180, 180) uint8

        for i in range(good_imgs.size(0)):
            if len(saved_images) < n_target:              # don't overshoot target
                saved_images.append(
                    Image.fromarray(good_imgs[i].numpy(), mode="L")
                )

    stats = {
        "target"          : n_target,
        "saved"           : len(saved_images),
        "total_generated" : total_generated,
        "total_passed"    : total_passed,
        "total_failed"    : total_failed,
        "pass_rate"       : total_passed / total_generated if total_generated > 0 else 0,
        "hit_limit"       : total_generated >= max_total_gen and len(saved_images) < n_target,
    }
    return saved_images, stats


def print_plan(plan: dict, out_dir: Path, threshold: float, max_attempts: int):
    print("\nGeneration plan:")
    print(f"  {'Joint':<6}  {'KL':<4}  {'Target':>7}")
    print(f"  {'-'*6}  {'-'*4}  {'-'*7}")
    total = 0
    for (joint, kl), n in plan.items():
        print(f"  {joint:<6}  KL={kl}  {n:>7} images")
        total += n
    print(f"  {'':20}  -------")
    print(f"  {'Total':<20}  {total:>7} images")
    print(f"\n  Filter threshold  : D(fake) >= {threshold}")
    print(f"  Max attempts      : {max_attempts}x per target count")
    print(f"  Output folder     : {out_dir}/\n")
    return total


def run(plan, g_path, d_path, out_dir, threshold, max_attempts, seed):
    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Load models ───────────────────────────────────────────────────────────
    G, D  = load_models(g_path, d_path)
    total = print_plan(plan, out_dir, threshold, max_attempts)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate, filter, save ────────────────────────────────────────────────
    saved_so_far = 0
    all_stats    = []

    for (joint, kl), n_target in plan.items():
        images, stats = generate_and_filter(
            G, D,
            joint_type   = joint,
            kl_score     = kl,
            n_target     = n_target,
            threshold    = threshold,
            max_attempts = max_attempts,
        )

        # Save images to disk
        for i, img in enumerate(images, start=1):
            filename = out_dir / f"{joint.upper()}_KL{kl}_{i:03d}.png"
            img.save(filename)

        saved_so_far += stats["saved"]
        all_stats.append(((joint, kl), stats))

        warn = "  !! INCOMPLETE" if stats["hit_limit"] else ""
        print(
            f"  [{saved_so_far:>4}/{total}]  "
            f"{joint} KL={kl}  ->  "
            f"{stats['saved']}/{n_target} saved  |  "
            f"generated {stats['total_generated']}  "
            f"(pass rate: {stats['pass_rate']*100:.1f}%)"
            f"{warn}"
        )

    # ── Final summary ─────────────────────────────────────────────────────────
    total_gen  = sum(s["total_generated"] for _, s in all_stats)
    total_save = sum(s["saved"]           for _, s in all_stats)
    overall    = total_save / total_gen * 100 if total_gen > 0 else 0

    print(f"\n{'='*60}")
    print(f"  Total generated : {total_gen}")
    print(f"  Total saved     : {total_save}  (overall pass rate: {overall:.1f}%)")
    print(f"  Total discarded : {total_gen - total_save}")
    print(f"  Output folder   : {out_dir}/")

    incomplete = [(jk, s) for jk, s in all_stats if s["hit_limit"]]
    if incomplete:
        print(f"\n  WARNING: {len(incomplete)} class(es) did not reach their target.")
        print(f"  Try lowering --threshold or raising --max-attempts:")
        for (joint, kl), s in incomplete:
            print(f"    {joint} KL={kl}: only saved {s['saved']}/{s['target']}")
    else:
        print(f"\n  All targets met!")
    print(f"{'='*60}")


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
        seed         = args.seed,
    )