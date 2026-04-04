"""
Step 2 v3: Train improved diffusion model (Fixed version).

Key improvements over v2:
- Min-SNR loss weighting (better detail learning)
- Proper latent scaling from VAE
- Lower EMA decay (0.995 vs 0.9999)
- Better class balancing with focal-style approach
- Validation generates images to track quality visually

Usage:
    python train_diffusion_v2.py --data_dir data/ --vae_ckpt checkpoints/vae_best.pt --epochs 500
"""

import os
import argparse
import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from dataset import HandJointDataset, create_dataloaders, load_patient_split
from models import VAE, ImprovedConditionalUNet, DiffusionScheduler, EMA


def create_balanced_loader(csv_path, image_dir, split_path, joint_types=['DIP'],
                           img_size=128, batch_size=64, num_workers=4):
    """
    Create a dataloader with class-balanced sampling.
    [IMPROVED] Better weighting strategy: square-root inverse frequency + explicit boost.
    """
    split = load_patient_split(split_path)

    train_dataset = HandJointDataset(
        csv_path, image_dir,
        patient_ids=split['train'],
        joint_types=joint_types,
        img_size=img_size, augment=True
    )

    kl_grades = train_dataset.data['kl_grade'].values.astype(int)
    class_counts = np.bincount(kl_grades, minlength=5).astype(float)
    class_counts = np.maximum(class_counts, 1)

    # [IMPROVED] Square-root inverse frequency is more stable than pure inverse
    class_weights = 1.0 / np.sqrt(class_counts)
    # Extra boost for KL3 and KL4 (our target classes)
    class_weights[3] *= 3.0
    class_weights[4] *= 5.0

    sample_weights = torch.tensor([class_weights[kl] for kl in kl_grades],
                                  dtype=torch.float)

    sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    print(f"\nClass-balanced training (sqrt inverse + boost):")
    for kl in range(5):
        count = (kl_grades == kl).sum()
        print(f"  KL{kl}: {count} images, weight={class_weights[kl]:.4f}")

    return train_loader, split


def train_one_epoch(model, vae, scheduler, loader, optimizer, device,
                    cfg_dropout_prob=0.15, use_min_snr=True, snr_gamma=5.0):
    """
    Train one epoch with:
    - Classifier-free guidance dropout
    - [NEW] Min-SNR loss weighting
    """
    model.train()
    vae.eval()
    total_loss = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        images = batch['image'].to(device)
        kl_grades = batch['kl_grade'].to(device)
        B = images.shape[0]

        with torch.no_grad():
            z_0 = vae.encode_to_latent(images)  # Now includes latent scaling

        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)
        z_t, noise = scheduler.add_noise(z_0, t)

        # CFG dropout
        drop_mask = torch.rand(B, device=device) < cfg_dropout_prob
        kl_input = kl_grades.clone()
        kl_input[drop_mask] = 5

        noise_pred = model(z_t, t, kl_input)

        # [NEW] Min-SNR weighted loss
        if use_min_snr:
            # Per-sample MSE loss
            per_sample_loss = F.mse_loss(noise_pred, noise, reduction='none')
            per_sample_loss = per_sample_loss.mean(dim=[1, 2, 3])  # [B]

            # Apply min-SNR weights
            snr_weights = scheduler.get_min_snr_weights(t, gamma=snr_gamma)
            loss = (per_sample_loss * snr_weights).mean()
        else:
            loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, vae, scheduler, loader, device):
    model.eval()
    vae.eval()
    total_loss = 0

    for batch in loader:
        images = batch['image'].to(device)
        kl_grades = batch['kl_grade'].to(device)
        B = images.shape[0]

        z_0 = vae.encode_to_latent(images)
        t = torch.randint(0, scheduler.num_timesteps, (B,), device=device)
        z_t, noise = scheduler.add_noise(z_0, t)
        noise_pred = model(z_t, t, kl_grades)
        loss = F.mse_loss(noise_pred, noise)
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def generate_samples(model, vae, scheduler, device, num_per_class=4,
                     cfg_scale=3.0, num_steps=50):
    """Generate sample images for each KL grade."""
    model.eval()
    vae.eval()
    all_images = []

    for kl in range(5):
        kl_label = torch.full((num_per_class,), kl, device=device, dtype=torch.long)
        z = scheduler.ddim_sample(
            model, shape=(num_per_class, 4, 16, 16),
            kl_grade=kl_label, num_steps=num_steps,
            cfg_scale=cfg_scale, device=device
        )
        images = vae.decode_from_latent(z)
        images = ((images + 1) / 2).clamp(0, 1)
        all_images.append(images)

    return all_images


def save_generation_grid(all_images, save_path, num_per_class=4):
    """Save grid: rows=KL grades, cols=samples."""
    fig, axes = plt.subplots(5, num_per_class, figsize=(2*num_per_class, 10))
    for kl in range(5):
        for j in range(num_per_class):
            axes[kl, j].imshow(all_images[kl][j, 0].cpu(), cmap='gray')
            if j == 0:
                axes[kl, j].set_ylabel(f'KL {kl}', fontsize=12)
            axes[kl, j].axis('off')
    plt.suptitle('Generated Samples by KL Grade', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--vae_ckpt', type=str, default='checkpoints/vae_best.pt')
    parser.add_argument('--joint_types', nargs='+', default=['DIP'])
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)  # [CHANGED] more epochs
    parser.add_argument('--lr', type=float, default=2e-4)   # [CHANGED] slightly higher LR
    parser.add_argument('--warmup_epochs', type=int, default=10)
    parser.add_argument('--base_ch', type=int, default=128)
    parser.add_argument('--num_timesteps', type=int, default=1000)
    parser.add_argument('--schedule', type=str, default='cosine',
                        choices=['cosine', 'linear'])
    parser.add_argument('--cfg_dropout', type=float, default=0.15)
    parser.add_argument('--cfg_scale', type=float, default=3.0)
    parser.add_argument('--ema_decay', type=float, default=0.995)  # [FIXED]
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--use_min_snr', action='store_true', default=True)  # [NEW]
    parser.add_argument('--snr_gamma', type=float, default=5.0)  # [NEW]
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--log_dir', type=str, default='logs/diffusion_v2')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load frozen VAE
    print("Loading VAE...")
    vae_ckpt = torch.load(args.vae_ckpt, map_location=device, weights_only=False)
    vae_args = vae_ckpt['args']
    vae = VAE(
        in_channels=1,
        latent_channels=vae_args.get('latent_channels', 4),
        base_ch=vae_args.get('base_ch', 64)
    ).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    # [NEW] Verify latent scale factor is set
    print(f"  VAE loaded (epoch {vae_ckpt['epoch']}, val_loss={vae_ckpt['val_loss']:.4f})")
    print(f"  Latent scale factor: {vae.latent_scale_factor.item():.4f}")
    if vae.latent_scale_factor.item() == 1.0:
        print("  WARNING: latent_scale_factor is 1.0. "
              "Run train_vae.py first to compute it, or it will be auto-computed now.")

    # Data with class-balanced sampling
    csv_path = os.path.join(args.data_dir, 'hand_long_clean2.csv')
    image_dir = os.path.join(args.data_dir, 'images')

    # Create standard loaders for val
    _, val_loader, _test_loader = create_dataloaders(
        csv_path, image_dir,
        joint_types=args.joint_types,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Create balanced training loader
    split_dir = os.path.dirname(csv_path) or '.'
    split_path = os.path.join(split_dir, f'split_{"_".join(args.joint_types)}_42.json')

    train_loader, _ = create_balanced_loader(
        csv_path, image_dir, split_path,
        joint_types=args.joint_types,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # [NEW] Auto-compute latent scale if not set
    if vae.latent_scale_factor.item() == 1.0:
        print("\nAuto-computing latent scale factor from training data...")
        vae.compute_latent_scale(train_loader, device)

    # Model
    model = ImprovedConditionalUNet(
        in_channels=vae_args.get('latent_channels', 4),
        base_ch=args.base_ch,
        num_kl_classes=5,
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Improved UNet parameters: {num_params:,}")

    # EMA with lower decay
    ema = EMA(model, decay=args.ema_decay)

    # Scheduler
    scheduler = DiffusionScheduler(
        num_timesteps=args.num_timesteps,
        schedule=args.schedule,
        device=device
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Warmup + cosine LR schedule
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / max(args.warmup_epochs, 1)
        progress = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume if needed
    start_epoch = 1
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['val_loss']
        if 'ema_state_dict' in ckpt:
            ema.load_state_dict(ckpt['ema_state_dict'])
        print(f"  Resumed from epoch {ckpt['epoch']}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, vae, scheduler, train_loader, optimizer, device,
            cfg_dropout_prob=args.cfg_dropout,
            use_min_snr=args.use_min_snr,
            snr_gamma=args.snr_gamma
        )

        # Update EMA
        ema.update(model)

        val_loss = validate(model, vae, scheduler, val_loader, device)
        lr_scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr_scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'diffusion_v2_best.pt'))
            print(f"  -> Saved best model")

        # Generate comparison every 25 epochs
        if epoch % 25 == 0 or epoch == 1:
            # Generate with raw model
            raw_images = generate_samples(
                model, vae, scheduler, device,
                num_per_class=4, cfg_scale=args.cfg_scale
            )
            save_generation_grid(
                raw_images,
                os.path.join(args.log_dir, f'raw_epoch{epoch:03d}.png')
            )

            # Generate with EMA model
            ema_images = generate_samples(
                ema.shadow, vae, scheduler, device,
                num_per_class=4, cfg_scale=args.cfg_scale
            )
            save_generation_grid(
                ema_images,
                os.path.join(args.log_dir, f'ema_epoch{epoch:03d}.png')
            )

        # Periodic checkpoint (for resume)
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'diffusion_v2_epoch{epoch}.pt'))

    # Plot curves
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Diffusion v3 Training Curves')
    plt.legend()
    plt.savefig(os.path.join(args.log_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()
