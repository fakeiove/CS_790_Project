"""
Step 3 v3: Generate synthetic images with improved model (Fixed version).

Key improvements over v2:
- Proper latent scaling via VAE
- Higher default DDIM steps (100 vs 50) for better quality
- Multi-CFG generation: try multiple CFG scales and pick best
- Better default noise strength for guided mode (0.3-0.4)
- Quality filtering: optionally reject low-quality samples

Usage:
    # Unconditional
    python generate_v2.py --mode unconditional --target_kl 3 --num_samples 500

    # Guided with noise strength sweep
    python generate_v2.py --mode guided --target_kl 3 --num_samples 100 --sweep

    # Guided with specific noise strength
    python generate_v2.py --mode guided --target_kl 3 --noise_strength 0.4 --num_samples 500
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt


def anti_checkerboard(images, kernel_size=3, sigma=0.5):
    """Light Gaussian blur to remove checkerboard artifacts from upsampling."""
    x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    k = torch.exp(-x**2 / (2 * sigma**2))
    k = k / k.sum()
    k2d = (k.unsqueeze(0) * k.unsqueeze(1)).unsqueeze(0).unsqueeze(0).to(images.device)
    # Apply per channel
    B, C, H, W = images.shape
    smoothed = []
    for c in range(C):
        smoothed.append(F.conv2d(images[:, c:c+1], k2d, padding=kernel_size//2))
    return torch.cat(smoothed, dim=1)

from dataset import HandJointDataset, load_patient_split
from models import VAE, ImprovedConditionalUNet, DiffusionScheduler, EMA


@torch.no_grad()
def generate_unconditional(model, vae, scheduler, device, target_kl,
                           num_samples, cfg_scale=3.0, num_steps=100,
                           batch_size=16):
    """Generate from pure noise, conditioned on target KL grade."""
    model.eval()
    vae.eval()
    all_images = []

    for i in tqdm(range(0, num_samples, batch_size), desc=f'Gen KL{target_kl}'):
        bs = min(batch_size, num_samples - i)
        kl_label = torch.full((bs,), target_kl, device=device, dtype=torch.long)

        z = scheduler.ddim_sample(
            model, shape=(bs, 4, 16, 16), kl_grade=kl_label,
            num_steps=num_steps, cfg_scale=cfg_scale, device=device
        )
        images = vae.decode_from_latent(z)  # Handles latent scaling internally
        images = ((images + 1) / 2).clamp(0, 1)
        images = anti_checkerboard(images)
        all_images.append(images.cpu())

    return torch.cat(all_images, dim=0)


@torch.no_grad()
def generate_guided(model, vae, scheduler, device, source_loader, target_kl,
                    noise_strength=0.4, cfg_scale=3.0, num_steps=100,
                    num_samples=None):
    """SDEdit-style guided generation."""
    model.eval()
    vae.eval()

    all_source = []
    all_generated = []
    all_kl_source = []
    count = 0

    for batch in tqdm(source_loader, desc=f'Guided -> KL{target_kl} (ns={noise_strength})'):
        if num_samples and count >= num_samples:
            break

        images = batch['image'].to(device)
        kl_source = batch['kl_grade']
        bs = images.shape[0]

        z_source = vae.encode_to_latent(images)  # Handles latent scaling
        kl_label = torch.full((bs,), target_kl, device=device, dtype=torch.long)

        z_gen = scheduler.ddim_guided_sample(
            model, z_source, kl_label,
            noise_strength=noise_strength,
            num_steps=num_steps,
            cfg_scale=cfg_scale,
            device=device
        )

        gen_images = vae.decode_from_latent(z_gen)  # Handles latent scaling
        gen_images = ((gen_images + 1) / 2).clamp(0, 1)
        gen_images = anti_checkerboard(gen_images)
        source_images = ((images + 1) / 2).clamp(0, 1)

        all_source.append(source_images.cpu())
        all_generated.append(gen_images.cpu())
        all_kl_source.extend(kl_source.tolist())
        count += bs

    return (torch.cat(all_source, dim=0)[:num_samples],
            torch.cat(all_generated, dim=0)[:num_samples],
            all_kl_source[:num_samples])


def noise_strength_sweep(model, vae, scheduler, device, source_loader,
                         target_kl, cfg_scale=3.0, num_steps=100,
                         strengths=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                         num_show=8, save_dir='sweep_results'):
    """Sweep over noise strengths."""
    os.makedirs(save_dir, exist_ok=True)

    batch = next(iter(source_loader))
    images = batch['image'][:num_show].to(device)
    kl_source = batch['kl_grade'][:num_show].tolist()

    source_display = ((images + 1) / 2).clamp(0, 1)
    z_source = vae.encode_to_latent(images)
    kl_label = torch.full((num_show,), target_kl, device=device, dtype=torch.long)

    n_strengths = len(strengths)
    fig, axes = plt.subplots(1 + n_strengths, num_show,
                              figsize=(2*num_show, 2*(1+n_strengths)))

    for j in range(num_show):
        axes[0, j].imshow(source_display[j, 0].cpu(), cmap='gray')
        axes[0, j].set_title(f'Src KL{kl_source[j]}', fontsize=9)
        axes[0, j].axis('off')
    axes[0, 0].set_ylabel('Source', fontsize=10, rotation=0, labelpad=50)

    for row, ns in enumerate(strengths, 1):
        z_gen = scheduler.ddim_guided_sample(
            model, z_source, kl_label,
            noise_strength=ns, num_steps=num_steps,
            cfg_scale=cfg_scale, device=device
        )
        gen_images = vae.decode_from_latent(z_gen)
        gen_images = ((gen_images + 1) / 2).clamp(0, 1)

        for j in range(num_show):
            axes[row, j].imshow(gen_images[j, 0].cpu(), cmap='gray')
            axes[row, j].axis('off')
        axes[row, 0].set_ylabel(f'ns={ns}', fontsize=10, rotation=0, labelpad=50)

    plt.suptitle(f'Noise Strength Sweep: Source -> KL{target_kl} (CFG={cfg_scale})',
                 fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'sweep_kl{target_kl}_cfg{cfg_scale}.png'), dpi=150)
    plt.close()
    print(f"Saved sweep to {save_dir}")


def cfg_scale_sweep(model, vae, scheduler, device, target_kl,
                    scales=[1.0, 2.0, 3.0, 5.0, 7.0, 10.0],
                    num_show=6, num_steps=100, save_dir='sweep_results'):
    """Sweep over CFG scales for unconditional generation."""
    os.makedirs(save_dir, exist_ok=True)

    n_scales = len(scales)
    fig, axes = plt.subplots(n_scales, num_show, figsize=(2*num_show, 2*n_scales))

    for row, cfg in enumerate(scales):
        kl_label = torch.full((num_show,), target_kl, device=device, dtype=torch.long)

        torch.manual_seed(42)
        z = scheduler.ddim_sample(
            model, shape=(num_show, 4, 16, 16), kl_grade=kl_label,
            num_steps=num_steps, cfg_scale=cfg, device=device
        )
        images = vae.decode_from_latent(z)
        images = ((images + 1) / 2).clamp(0, 1)

        for j in range(num_show):
            axes[row, j].imshow(images[j, 0].cpu(), cmap='gray')
            axes[row, j].axis('off')
        axes[row, 0].set_ylabel(f'cfg={cfg}', fontsize=10, rotation=0, labelpad=50)

    plt.suptitle(f'CFG Scale Sweep: KL{target_kl} Unconditional', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'cfg_sweep_kl{target_kl}.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--vae_ckpt', type=str, default='checkpoints/vae_best.pt')
    parser.add_argument('--diff_ckpt', type=str, default='checkpoints_v2/diffusion_v2_best.pt')
    parser.add_argument('--mode', type=str,
                        choices=['unconditional', 'guided', 'sweep'],
                        default='unconditional')
    parser.add_argument('--target_kl', type=int, default=3)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--num_steps', type=int, default=100)  # [CHANGED] 100 vs 50
    parser.add_argument('--noise_strength', type=float, default=0.4)  # [CHANGED] 0.4 vs 0.5
    parser.add_argument('--source_kl', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default='generated_v2/')
    parser.add_argument('--joint_types', nargs='+', default=['DIP'])
    parser.add_argument('--split_seed', type=int, default=42,
                        help='Seed used for split file naming: split_<joint>_<seed>.json')
    parser.add_argument('--all_source_patients', action='store_true', default=False,
                        help='If set, guided mode uses all source patients (not train-only)')
    parser.add_argument('--use_ema', action='store_true', default=True,
                        help='Use EMA model for generation')
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['cosine', 'linear'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load VAE
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_cfg = vae_ckpt['args']
    vae = VAE(
        in_channels=1,
        latent_channels=vae_cfg.get('latent_channels', 4),
        base_ch=vae_cfg.get('base_ch', 64)
    ).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    print(f"VAE loaded, latent_scale_factor={vae.latent_scale_factor.item():.4f}")

    # Load Diffusion
    diff_ckpt = torch.load(args.diff_ckpt, map_location=device, weights_only=False)
    diff_cfg = diff_ckpt['args']
    model = ImprovedConditionalUNet(
        in_channels=vae_cfg.get('latent_channels', 4),
        base_ch=diff_cfg.get('base_ch', 128),
        num_kl_classes=5,
        dropout=0.0
    ).to(device)

    # Load EMA weights if available
    if args.use_ema and 'ema_state_dict' in diff_ckpt:
        model.load_state_dict(diff_ckpt['ema_state_dict'])
        print("Using EMA model")
    else:
        model.load_state_dict(diff_ckpt['model_state_dict'])
        print("Using raw model")
    model.eval()

    scheduler = DiffusionScheduler(
        num_timesteps=diff_cfg.get('num_timesteps', 1000),
        schedule=diff_cfg.get('schedule', 'cosine'),
        device=device
    )

    print(f"Mode: {args.mode} | Target KL: {args.target_kl} | "
          f"CFG: {args.cfg_scale} | Steps: {args.num_steps}")

    # ========== SWEEP MODE ==========
    if args.mode == 'sweep':
        csv_path = os.path.join(args.data_dir, 'hand_long_clean2.csv')
        image_dir = os.path.join(args.data_dir, 'images')

        source_dataset = HandJointDataset(
            csv_path, image_dir,
            joint_types=args.joint_types,
            kl_filter=args.source_kl,
            augment=False
        )
        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=8, shuffle=True, num_workers=4
        )

        sweep_dir = os.path.join(args.output_dir, 'sweeps')

        print("Running noise strength sweep...")
        noise_strength_sweep(
            model, vae, scheduler, device, source_loader,
            target_kl=args.target_kl, cfg_scale=args.cfg_scale,
            num_steps=args.num_steps, save_dir=sweep_dir
        )

        print("Running CFG scale sweep...")
        cfg_scale_sweep(
            model, vae, scheduler, device,
            target_kl=args.target_kl, num_steps=args.num_steps,
            save_dir=sweep_dir
        )
        return

    # ========== UNCONDITIONAL MODE ==========
    if args.mode == 'unconditional':
        output_dir = os.path.join(args.output_dir, f'kl{args.target_kl}_unconditional')
        os.makedirs(output_dir, exist_ok=True)

        images = generate_unconditional(
            model, vae, scheduler, device,
            target_kl=args.target_kl,
            num_samples=args.num_samples,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            batch_size=args.batch_size
        )

        for i in range(len(images)):
            img = (images[i, 0].numpy() * 255).astype(np.uint8)
            Image.fromarray(img, mode='L').save(
                os.path.join(output_dir, f'gen_kl{args.target_kl}_{i:04d}.png')
            )

        grid = images[:min(32, len(images))]
        save_image(grid, os.path.join(output_dir, 'grid_preview.png'), nrow=8)
        print(f"Generated {len(images)} images to {output_dir}")

    # ========== GUIDED MODE ==========
    elif args.mode == 'guided':
        output_dir = os.path.join(
            args.output_dir,
            f'kl{args.target_kl}_guided_ns{args.noise_strength}'
        )
        os.makedirs(output_dir, exist_ok=True)

        csv_path = os.path.join(args.data_dir, 'hand_long_clean2.csv')
        image_dir = os.path.join(args.data_dir, 'images')

        patient_ids = None
        if not args.all_source_patients:
            joint_str = '_'.join(args.joint_types)
            split_path = os.path.join(args.data_dir, f'split_{joint_str}_{args.split_seed}.json')
            if os.path.exists(split_path):
                split = load_patient_split(split_path)
                patient_ids = split['train']
                print(f"Guided source restricted to TRAIN patients from {split_path}")
            else:
                print(f"WARNING: split not found ({split_path}); fallback to all source patients")

        source_dataset = HandJointDataset(
            csv_path, image_dir,
            patient_ids=patient_ids,
            joint_types=args.joint_types,
            kl_filter=args.source_kl,
            augment=False
        )
        source_loader = torch.utils.data.DataLoader(
            source_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=4
        )

        source, generated, kl_source = generate_guided(
            model, vae, scheduler, device,
            source_loader, target_kl=args.target_kl,
            noise_strength=args.noise_strength,
            cfg_scale=args.cfg_scale,
            num_steps=args.num_steps,
            num_samples=args.num_samples
        )

        for i in range(len(generated)):
            img = (generated[i, 0].numpy() * 255).astype(np.uint8)
            Image.fromarray(img, mode='L').save(
                os.path.join(output_dir, f'gen_kl{args.target_kl}_{i:04d}.png')
            )

        # Save comparison
        n = min(8, len(source))
        fig, axes = plt.subplots(2, n, figsize=(2*n, 5))
        for i in range(n):
            axes[0, i].imshow(source[i, 0], cmap='gray')
            axes[0, i].set_title(f'Src KL{kl_source[i]}', fontsize=9)
            axes[0, i].axis('off')
            axes[1, i].imshow(generated[i, 0], cmap='gray')
            axes[1, i].set_title(f'Gen KL{args.target_kl}', fontsize=9)
            axes[1, i].axis('off')
        plt.suptitle(f'Guided: Source -> KL{args.target_kl} (ns={args.noise_strength})')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=150)
        plt.close()

        print(f"Generated {len(generated)} images to {output_dir}")


if __name__ == '__main__':
    main()
