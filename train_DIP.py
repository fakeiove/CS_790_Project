"""
Conditional GAN (cGAN) for Synthetic DIP X-Ray Image Generation
================================================================
Approach 1: DIP joints only, conditioned on KL score (0-4) only.
Joint type condition removed entirely since we only train on DIP.

Changes from previous version:
  - Dataset filtered to DIP joints only
  - emb_joint removed from Generator and Discriminator
  - input_dim: 100 + 16 = 116  (was 124)
  - label_proj: 16 → 180*180   (was 24 → 180*180)
  - Discriminator forward: only emb_kl, no emb_joint
  - make_weighted_sampler: weights by KL score only
  - save_sample_grid: no jt_t needed
  - generate_synthetic_images: no joint_type loop
  - WGAN-GP loss throughout
"""

import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE   = 180
LATENT_DIM = 100
N_KL       = 5       # KL scores: 0, 1, 2, 3, 4
EMB_KL     = 16      # embedding dimension for KL score
N_CRITIC = 5         # number of D updates per G update (WGAN-GP recommends 5)

BATCH_SIZE  = 64
NUM_EPOCHS  = 200
LR          = 1e-4   # same LR for both G and D (WGAN-GP standard)
LAMBDA_GP   = 10     # gradient penalty weight
SAVE_EVERY  = 10

IMAGE_DIR      = Path("Finger joints")
CSV_PATH       = Path("hand_long_clean2.csv")
OUTPUT_DIR     = Path("cgan_output_dip")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset  — DIP only, returns (image, kl_label) only
# ─────────────────────────────────────────────────────────────────────────────
class DIPDataset(Dataset):
    """
    Loads only DIP joint images.
    Returns (image_tensor, kl_label) — no joint_type label needed.
    """

    def __init__(self, csv_path: Path, image_dir: Path, transform=None):
        df = pd.read_csv(csv_path)

        # ── Keep DIP rows only ────────────────────────────────────────────────
        df = df[df["joint_type"].str.upper() == "DIP"].reset_index(drop=True)

        # ── Drop rows with missing KL score ───────────────────────────────────
        df = df.dropna(subset=["v00_KL"]).reset_index(drop=True)
        df["v00_KL"] = df["v00_KL"].astype(int)

        self.df        = df
        self.image_dir = image_dir
        self.transform = transform or self._default_transform()

        # Print class distribution so you can verify the filter worked
        counts = df["v00_KL"].value_counts().sort_index()
        print("DIP dataset — KL distribution:")
        for kl, count in counts.items():
            print(f"  KL={kl}: {count:,} images  ({count/len(df)*100:.1f}%)")
        print(f"  Total : {len(df):,} images\n")

    @staticmethod
    def _default_transform():
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.Grayscale(num_output_channels=1),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),            # [0,1] -> [-1,1]
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        patient_id = str(int(row["patient_id"]))
        joint      = str(row["joint"]).lower()    # e.g. "dip2"
        img_path   = self.image_dir / f"{patient_id}_{joint}.png"

        try:
            image = Image.open(img_path).convert("L")
        except FileNotFoundError:
            image = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))

        image    = self.transform(image)
        kl_label = torch.tensor(int(row["v00_KL"]), dtype=torch.long)

        return image, kl_label          # only 2 items, no jt_label


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Generator  — conditioned on KL score only
# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Input : z (100) + emb_kl (16)  ->  total input_dim = 116
    Output: 1 x 180 x 180 image

    Spatial progression:
      Linear -> reshape  :  5 x  5  (512 ch)
      ConvTranspose2d    : 10 x 10  (256 ch)
      ConvTranspose2d    : 20 x 20  (128 ch)
      ConvTranspose2d    : 40 x 40  ( 64 ch)
      ConvTranspose2d    : 80 x 80  ( 32 ch)
      Upsample(180)      :180 x180  bilinear
      Conv2d + Tanh      :180 x180  (  1 ch)
    """

    def __init__(self):
        super().__init__()
        self.emb_kl = nn.Embedding(N_KL, EMB_KL)

        input_dim = LATENT_DIM + EMB_KL          # 100 + 16 = 116

        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512 * 5 * 5),
            nn.BatchNorm1d(512 * 5 * 5),
            nn.ReLU(True),
        )

        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )

        self.up1 = up_block(512, 256)
        self.up2 = up_block(256, 128)
        self.up3 = up_block(128,  64)
        self.up4 = up_block( 64,  32)

        self.to_img = nn.Sequential(
            nn.Upsample(size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, kl_label):
        e_kl = self.emb_kl(kl_label)                    # (B, 16)
        x    = torch.cat([z, e_kl], dim=1)              # (B, 116)

        x = self.proj(x)
        x = x.view(-1, 512, 5, 5)

        x = self.up1(x)                                  # (B, 256, 10, 10)
        x = self.up2(x)                                  # (B, 128, 20, 20)
        x = self.up3(x)                                  # (B,  64, 40, 40)
        x = self.up4(x)                                  # (B,  32, 80, 80)

        return self.to_img(x)                            # (B,   1,180,180)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Discriminator  — conditioned on KL score only
# ─────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    Input : 1x180x180 image + KL label -> raw score (no sigmoid, WGAN-GP)

    KL embedding projected to 1-channel 180x180 spatial map,
    concatenated with image -> 2 input channels total.
    """

    def __init__(self):
        super().__init__()
        self.emb_kl     = nn.Embedding(N_KL, EMB_KL)
        self.label_proj = nn.Linear(EMB_KL, IMG_SIZE * IMG_SIZE)  # 16 -> 180*180

        def down_block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch,
                                kernel_size=4, stride=2, padding=1, bias=False)]
            if bn:
                layers.append(nn.InstanceNorm2d(out_ch, affine=True))  # Repllace BatchNorm with InstanceNorm
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.conv_blocks = nn.Sequential(
            down_block(  2,  64, bn=False),   # 180 ->  90
            down_block( 64, 128),              #  90 ->  45
            down_block(128, 256),              #  45 ->  22
            down_block(256, 512),              #  22 ->  11
            down_block(512, 512),              #  11 ->   5
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B, 512, 1, 1)
            nn.Flatten(),                      # (B, 512)
            nn.Linear(512, 1),                 # raw score, no Sigmoid
        )

    def forward(self, img, kl_label):
        B = img.size(0)

        e_kl  = self.emb_kl(kl_label)                   # (B, 16)
        label = self.label_proj(e_kl)                    # (B, 180*180)
        label = label.view(B, 1, IMG_SIZE, IMG_SIZE)     # (B, 1, 180, 180)

        x = torch.cat([img, label], dim=1)               # (B, 2, 180, 180)
        x = self.conv_blocks(x)
        return self.classifier(x)                        # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Weight initialisation
# ─────────────────────────────────────────────────────────────────────────────
def weights_init(m):
    classname = m.__class__.__name__
    if "Conv" in classname and hasattr(m, "weight") and m.weight is not None:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Visualisation  — one row per KL score, no joint_type needed
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def save_sample_grid(generator, epoch, n_per_kl=4):
    generator.eval()
    all_imgs = []

    for kl in range(N_KL):
        z    = torch.randn(n_per_kl, LATENT_DIM, device=DEVICE)
        kl_t = torch.full((n_per_kl,), kl, dtype=torch.long, device=DEVICE)
        imgs = generator(z, kl_t)                        # no jt_t
        all_imgs.append(imgs.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)

    grid = vutils.make_grid(
        all_imgs, nrow=n_per_kl, normalize=True, value_range=(-1, 1), padding=4
    )

    save_path = OUTPUT_DIR / f"samples_epoch_{epoch:04d}.png"
    vutils.save_image(grid, save_path)
    print(f"  [Saved sample grid -> {save_path}]")

    generator.train()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Weighted sampler  — by KL score only
# ─────────────────────────────────────────────────────────────────────────────
def make_weighted_sampler(dataset):
    kl_labels = [int(row["v00_KL"]) for _, row in dataset.df.iterrows()]
    counts    = Counter(kl_labels)
    weights   = torch.tensor(
        [1.0 / counts[kl] for kl in kl_labels], dtype=torch.float
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  WGAN-GP gradient penalty
# ─────────────────────────────────────────────────────────────────────────────
def gradient_penalty(D, real_imgs, fake_imgs, kl_labels):
    B      = real_imgs.size(0)
    alpha  = torch.rand(B, 1, 1, 1, device=DEVICE)
    interp = (alpha * real_imgs + (1 - alpha) * fake_imgs.detach()).requires_grad_(True)

    d_interp  = D(interp, kl_labels)                     # no jt_labels
    gradients = torch.autograd.grad(
        outputs      = d_interp,
        inputs       = interp,
        grad_outputs = torch.ones_like(d_interp),
        create_graph = True,
        retain_graph = True,
    )[0]

    gradients = gradients.view(B, -1)
    penalty   = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_cgan():
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset    = DIPDataset(CSV_PATH, IMAGE_DIR)
    sampler    = make_weighted_sampler(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size  = BATCH_SIZE,
        sampler     = sampler,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
    )
    print(f"Batches/epoch: {len(dataloader)}")

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)
    print(f"Generator params    : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}\n")

    # ── Optimisers (WGAN-GP: betas=(0.0, 0.9)) ───────────────────────────────
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(0.0, 0.9))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(0.0, 0.9))

    history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_d, epoch_g, epoch_dr, epoch_df = [], [], [], []

        for real_imgs, kl_labels in dataloader:          # only 2 items unpacked
            B         = real_imgs.size(0)
            real_imgs = real_imgs.to(DEVICE)
            kl_labels = kl_labels.to(DEVICE)

            # ── (A) Update Discriminator ─────────────────────────────
            for _ in range(N_CRITIC):
                opt_D.zero_grad()
                z         = torch.randn(B, LATENT_DIM, device=DEVICE)
                fake_imgs = G(z, kl_labels).detach()
                d_real    = D(real_imgs, kl_labels).mean()
                d_fake    = D(fake_imgs, kl_labels).mean()
                gp        = gradient_penalty(D, real_imgs, fake_imgs, kl_labels)
                d_loss    = -d_real + d_fake + LAMBDA_GP * gp
                d_loss.backward()
                opt_D.step()

            # ── (B) Update Generator ─────────────────────
            opt_G.zero_grad()
            z         = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z, kl_labels)
            g_loss    = -D(fake_imgs, kl_labels).mean()
            g_loss.backward()
            opt_G.step()

            epoch_d.append(d_loss.item())
            epoch_g.append(g_loss.item())
            epoch_dr.append(d_real.item())
            epoch_df.append(d_fake.item())

        d_avg  = np.mean(epoch_d)
        g_avg  = np.mean(epoch_g)
        dr_avg = np.mean(epoch_dr)
        df_avg = np.mean(epoch_df)

        history["d_loss"].append(d_avg)
        history["g_loss"].append(g_avg)
        history["d_real"].append(dr_avg)
        history["d_fake"].append(df_avg)

        print(
            f"Epoch [{epoch:4d}/{NUM_EPOCHS}]  "
            f"D_loss: {d_avg:.4f}  G_loss: {g_avg:.4f}  "
            f"D(real): {dr_avg:.3f}  D(fake): {df_avg:.3f}"
        )

        if epoch % SAVE_EVERY == 0 or epoch == 1:
            save_sample_grid(G, epoch)
            torch.save(G.state_dict(), CHECKPOINT_DIR / f"G_epoch_{epoch:04d}.pth")
            torch.save(D.state_dict(), CHECKPOINT_DIR / f"D_epoch_{epoch:04d}.pth")

    # ── Training curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["d_loss"], label="D loss")
    axes[0].plot(history["g_loss"], label="G loss")
    axes[0].set_title("Generator & Discriminator Loss (WGAN-GP)")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["d_real"], label="D(real)")
    axes[1].plot(history["d_fake"], label="D(fake)")
    axes[1].set_title("Average Critic Scores")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)

    return G, D, history


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Generation utility
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_synthetic_images(generator, n_per_kl=200, kl_scores=None):
    """
    Generate n_per_kl DIP images for each KL score.
    Returns dict: kl -> list of PIL images
    """
    generator.eval()
    kl_scores = kl_scores or list(range(N_KL))
    results   = {}

    for kl in kl_scores:
        imgs_out  = []
        remaining = n_per_kl

        while remaining > 0:
            bs   = min(BATCH_SIZE, remaining)
            z    = torch.randn(bs, LATENT_DIM, device=DEVICE)
            kl_t = torch.full((bs,), kl, dtype=torch.long, device=DEVICE)
            out  = generator(z, kl_t).cpu()
            out  = (out * 0.5 + 0.5).clamp(0, 1)
            out  = (out * 255).byte().squeeze(1).numpy()
            imgs_out.extend([Image.fromarray(o) for o in out])
            remaining -= bs

        results[kl] = imgs_out
        print(f"  KL={kl}  ->  {len(imgs_out)} DIP images generated")

    generator.train()
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G, D, history = train_cgan()