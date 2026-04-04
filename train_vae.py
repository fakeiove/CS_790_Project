"""
Step 1: Train VAE (Fixed version)

Key improvements:
- Perceptual loss (LPIPS-like) for sharper textures instead of pure MSE
- KL weight warmup: gradually increase KL weight to prevent early training collapse
- SSIM metric tracking
- Better learning rate schedule

Usage:
    python train_vae.py --data_dir data/ --epochs 150 --batch_size 64
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt
import math

from dataset import HandJointDataset, create_dataloaders
from models import VAE, vae_loss


# ============================================================
# Perceptual Loss (lightweight, no extra dependencies needed)
# ============================================================

class PerceptualLoss(nn.Module):
    """
    Lightweight perceptual loss using VGG16 features.
    Helps produce sharper, more detailed reconstructions than pure MSE.

    For grayscale images: replicates to 3 channels before VGG.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features

        # Use features from layers: relu1_2, relu2_2, relu3_3
        self.blocks = nn.ModuleList([
            vgg[:4],   # relu1_2
            vgg[4:9],  # relu2_2
            vgg[9:16], # relu3_3
        ])

        for p in self.parameters():
            p.requires_grad = False

        self.to(device)
        self.eval()

        # Feature weights (deeper features weighted more)
        self.weights = [1.0, 1.0, 1.0]

    @torch.no_grad()
    def forward(self, pred, target):
        """
        Args:
            pred, target: [B, 1, H, W] in range [-1, 1]
        Returns:
            perceptual loss (scalar)
        """
        # Grayscale -> 3-channel, normalize to ImageNet range
        pred_rgb = pred.repeat(1, 3, 1, 1)
        target_rgb = target.repeat(1, 3, 1, 1)

        # Scale from [-1,1] to [0,1], then ImageNet normalize
        pred_rgb = (pred_rgb + 1) / 2
        target_rgb = (target_rgb + 1) / 2

        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
        pred_rgb = (pred_rgb - mean) / std
        target_rgb = (target_rgb - mean) / std

        loss = 0
        x_pred = pred_rgb
        x_target = target_rgb

        for block, weight in zip(self.blocks, self.weights):
            x_pred = block(x_pred)
            x_target = block(x_target)
            loss += weight * F.mse_loss(x_pred, x_target)

        return loss


# ============================================================
# SSIM metric
# ============================================================

def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM between two batches of images. Range: [0, 1], higher is better."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    window = g.unsqueeze(0) * g.unsqueeze(1)
    window = window.unsqueeze(0).unsqueeze(0).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2)
    mu2 = F.conv2d(img2, window, padding=window_size // 2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 ** 2, window, padding=window_size // 2) - mu1_sq
    sigma2_sq = F.conv2d(img2 ** 2, window, padding=window_size // 2) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2) - mu12

    ssim = ((2 * mu12 + C1) * (2 * sigma12 + C2)) / \
           ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim.mean().item()


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, loader, optimizer, device, kl_weight,
                    perceptual_loss_fn=None, perc_weight=0.1):
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_perc = 0

    for batch in tqdm(loader, desc='Training', leave=False):
        images = batch['image'].to(device)

        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_weight)

        # [NEW] Add perceptual loss for sharper textures
        perc_loss = torch.tensor(0.0, device=device)
        if perceptual_loss_fn is not None:
            with torch.no_grad():
                perc_loss = perceptual_loss_fn(recon, images)
            # We need gradients through recon, so recompute with grad
            perc_loss = _compute_perceptual_with_grad(
                perceptual_loss_fn, recon, images)
            loss = loss + perc_weight * perc_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_perc += perc_loss.item() if isinstance(perc_loss, torch.Tensor) else perc_loss

    n = len(loader)
    return total_loss / n, total_recon / n, total_kl / n, total_perc / n


def _compute_perceptual_with_grad(perc_fn, pred, target):
    """Compute perceptual loss allowing gradients through pred."""
    # Grayscale -> 3-channel
    pred_rgb = pred.repeat(1, 3, 1, 1)
    target_rgb = target.repeat(1, 3, 1, 1)

    pred_rgb = (pred_rgb + 1) / 2
    target_rgb = (target_rgb + 1) / 2

    mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)
    pred_rgb = (pred_rgb - mean) / std
    target_rgb = (target_rgb - mean) / std

    loss = 0
    x_pred = pred_rgb
    x_target = target_rgb.detach()

    for block, weight in zip(perc_fn.blocks, perc_fn.weights):
        x_pred = block(x_pred)
        with torch.no_grad():
            x_target = block(x_target)
        loss += weight * F.mse_loss(x_pred, x_target)

    return loss


@torch.no_grad()
def validate(model, loader, device, kl_weight):
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_ssim = 0
    n_batches = 0

    for batch in loader:
        images = batch['image'].to(device)
        recon, mu, logvar = model(images)
        loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, kl_weight)

        # Compute SSIM on [0,1] range
        img_01 = (images + 1) / 2
        recon_01 = (recon + 1) / 2
        ssim = compute_ssim(img_01, recon_01)

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_ssim += ssim
        n_batches += 1

    return (total_loss / n_batches, total_recon / n_batches,
            total_kl / n_batches, total_ssim / n_batches)


@torch.no_grad()
def save_reconstructions(model, loader, device, save_path, num_samples=8):
    """Save side-by-side comparison of original vs reconstructed images."""
    model.eval()
    batch = next(iter(loader))
    images = batch['image'][:num_samples].to(device)
    kl_grades = batch['kl_grade'][:num_samples]

    recon, _, _ = model(images)

    # Denormalize: [-1,1] -> [0,1]
    images = (images + 1) / 2
    recon = (recon + 1) / 2

    # Interleave original and reconstruction
    comparison = torch.cat([images, recon], dim=0)
    save_image(comparison, save_path, nrow=num_samples, padding=2)

    # Also save with KL labels
    fig, axes = plt.subplots(2, num_samples, figsize=(2*num_samples, 4))
    for i in range(num_samples):
        axes[0, i].imshow(images[i, 0].cpu(), cmap='gray')
        axes[0, i].set_title(f'KL{kl_grades[i]}', fontsize=10)
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i, 0].cpu(), cmap='gray')
        axes[1, i].set_title('Recon', fontsize=10)
        axes[1, i].axis('off')
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_labeled.png'), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--joint_types', nargs='+', default=['DIP'])
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)  # [CHANGED] more epochs
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--kl_weight', type=float, default=1e-4,
                        help='Final KL divergence weight (with warmup).')
    parser.add_argument('--kl_warmup_epochs', type=int, default=20,
                        help='[NEW] Gradually increase KL weight over this many epochs.')
    parser.add_argument('--perc_weight', type=float, default=0.1,
                        help='[NEW] Perceptual loss weight.')
    parser.add_argument('--use_perceptual', action='store_true', default=True,
                        help='[NEW] Use perceptual loss for sharper textures.')
    parser.add_argument('--latent_channels', type=int, default=4)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs/vae')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Data
    csv_path = os.path.join(args.data_dir, 'hand_long_clean2.csv')
    image_dir = os.path.join(args.data_dir, 'images')

    train_loader, val_loader, _test_loader = create_dataloaders(
        csv_path, image_dir,
        joint_types=args.joint_types,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Model
    model = VAE(
        in_channels=1,
        latent_channels=args.latent_channels,
        base_ch=args.base_ch
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"VAE parameters: {num_params:,}")

    # [NEW] Perceptual loss
    perceptual_loss_fn = None
    if args.use_perceptual:
        perceptual_loss_fn = PerceptualLoss(device)
        print("Using perceptual loss")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_ssims = []

    for epoch in range(1, args.epochs + 1):
        # [NEW] KL weight warmup
        if epoch <= args.kl_warmup_epochs:
            current_kl_weight = args.kl_weight * (epoch / args.kl_warmup_epochs)
        else:
            current_kl_weight = args.kl_weight

        train_loss, train_recon, train_kl, train_perc = train_one_epoch(
            model, train_loader, optimizer, device, current_kl_weight,
            perceptual_loss_fn=perceptual_loss_fn, perc_weight=args.perc_weight
        )
        val_loss, val_recon, val_kl, val_ssim = validate(
            model, val_loader, device, current_kl_weight
        )
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_ssims.append(val_ssim)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"Train: {train_loss:.4f} (recon:{train_recon:.4f}, kl:{train_kl:.2f}, perc:{train_perc:.4f}) | "
              f"Val: {val_loss:.4f} (recon:{val_recon:.4f}, SSIM:{val_ssim:.4f}) | "
              f"KL_w: {current_kl_weight:.6f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ssim': val_ssim,
                'args': vars(args)
            }, os.path.join(args.save_dir, 'vae_best.pt'))
            print(f"  -> Saved best model (val_loss={val_loss:.4f}, SSIM={val_ssim:.4f})")

        # Save reconstructions
        if epoch % 10 == 0 or epoch == 1:
            save_reconstructions(
                model, val_loader, device,
                os.path.join(args.log_dir, f'recon_epoch{epoch:03d}.png')
            )

        # Save periodic checkpoints
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_ssim': val_ssim,
                'args': vars(args)
            }, os.path.join(args.save_dir, f'vae_epoch{epoch}.pt'))

    # [NEW] After training: compute and save latent scale factor
    print("\nComputing latent scale factor...")
    model.eval()
    scale = model.compute_latent_scale(train_loader, device)

    # Re-save best model with scale factor baked into state_dict
    best_ckpt = torch.load(os.path.join(args.save_dir, 'vae_best.pt'), map_location=device)
    best_ckpt['latent_scale_factor'] = scale.item()
    # Update the latent_scale_factor in the state dict directly
    state_dict = best_ckpt['model_state_dict']
    state_dict['latent_scale_factor'] = scale
    best_ckpt['model_state_dict'] = state_dict
    torch.save(best_ckpt, os.path.join(args.save_dir, 'vae_best.pt'))
    print(f"Saved latent_scale_factor={scale.item():.4f} into vae_best.pt")

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('VAE Training Curves')
    ax1.legend()

    ax2.plot(val_ssims, label='Val SSIM', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('SSIM')
    ax2.set_title('Reconstruction Quality (SSIM)')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, 'training_curves.png'), dpi=150)
    plt.close()

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {os.path.join(args.save_dir, 'vae_best.pt')}")


if __name__ == '__main__':
    main()
