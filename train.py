"""
Conditional GAN (cGAN) for Synthetic X-Ray Finger Joint Image Generation
=========================================================================
Generates grayscale 1×180×180 X-ray images conditioned on:
  - KL score  (0–4)
  - Joint type (DIP=0, PIP=1, MCP=2)

180×180 Dimension Math
-----------------------
180 is NOT a power of 2, so we cannot simply stack standard 2× upsampling
layers.  The strategy used here is:

  Generator (noise → image)
  ──────────────────────────
  We start from a 5×5 spatial map and grow it with ConvTranspose2d layers,
  then finish with a bilinear Upsample to hit exactly 180×180.

  Layer-by-layer spatial sizes (H = W):
    Linear → reshape      :   5 ×  5   (512 ch)
    ConvTranspose2d s=2   :  10 × 10   (256 ch)   formula: (5-1)*2+4 = 10 ✓
    ConvTranspose2d s=2   :  20 × 20   (128 ch)   formula: (10-1)*2+4 = 20 ✓
    ConvTranspose2d s=2   :  40 × 40   ( 64 ch)   formula: (20-1)*2+4 = 40 ✓
    ConvTranspose2d s=2   :  80 × 80   ( 32 ch)   formula: (40-1)*2+4 = 80 ✓
    Upsample(180,180)     : 180 ×180   bilinear, align_corners=False
    Conv2d (3×3, pad=1)   : 180 ×180   ( 1 ch) + Tanh

  ConvTranspose2d output size formula:
    H_out = (H_in − 1) × stride − 2×padding + kernel_size
  With kernel=4, stride=2, padding=1:
    H_out = (H_in − 1) × 2 + 4 − 2 = 2 × H_in ✓  (clean doubling)

  The final Upsample(180) step stretches 80→180 (factor 2.25×) without
  introducing checkerboard artifacts, then a learned Conv2d refines details.

  Discriminator (image → real/fake)
  ───────────────────────────────────
  Downsamples 180→90→45→22→11→5 via Conv2d stride=2 layers, then
  GlobalAvgPool + Linear → sigmoid.
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports
# ─────────────────────────────────────────────────────────────────────────────
import os
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torch.utils.data import WeightedRandomSampler
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
IMG_SIZE      = 180
LATENT_DIM    = 100
N_KL          = 5          # KL scores: 0,1,2,3,4
N_JOINT       = 3          # DIP=0, PIP=1, MCP=2
EMB_KL        = 16         # embedding dimension for KL score
EMB_JOINT     = 8          # embedding dimension for joint type

BATCH_SIZE    = 64
NUM_EPOCHS    = 200
LR            = 2e-4
BETA1         = 0.5
SAVE_EVERY    = 10         # save image grid every N epochs

IMAGE_DIR     = Path("Finger joints")   # folder containing .png files
CSV_PATH      = Path("hand_long_clean2.csv")
OUTPUT_DIR    = Path("cgan_output")
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 3.  Dataset
# ─────────────────────────────────────────────────────────────────────────────
JOINT_TYPE_MAP = {"DIP": 0, "PIP": 1, "MCP": 2}

class FingerJointDataset(Dataset):
    """
    Returns (image_tensor, kl_label, joint_type_label) for each sample.

    Filename convention: "{patient_id}_{joint_code}.png"
    e.g.  "9000099_dip2.png"  →  patient_id=9000099, joint="dip2"
    The joint_type is the alphabetic prefix of the joint column
    (dip → DIP, pip → PIP, mcp → MCP).
    """

    def __init__(self, csv_path: Path, image_dir: Path, transform=None):
        df = pd.read_csv(csv_path)

        # ── Drop rows with missing KL score ──────────────────────────────────
        df = df.dropna(subset=["v00_KL"]).reset_index(drop=True)
        df["v00_KL"] = df["v00_KL"].astype(int)

        # ── Map joint_type to int ─────────────────────────────────────────────
        df["joint_type_id"] = (
            df["joint_type"].str.upper().map(JOINT_TYPE_MAP)
        )
        unknown = df["joint_type_id"].isna()
        if unknown.any():
            bad = df.loc[unknown, "joint_type"].unique().tolist()
            raise ValueError(f"Unknown joint_type values: {bad}")
        df["joint_type_id"] = df["joint_type_id"].astype(int)

        self.df         = df
        self.image_dir  = image_dir
        self.transform  = transform or self._default_transform()

    @staticmethod
    def _default_transform():
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.Grayscale(num_output_channels=1),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(
                degrees=10,              # small rotation
                translate=(0.05, 0.05),  # small shift
                scale=(0.95, 1.05),      # slight zoom
            ),
            T.ToTensor(),
            T.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row         = self.df.iloc[idx]
        patient_id  = str(int(row["patient_id"]))
        joint       = str(row["joint"]).lower()           # e.g. "dip2"
        filename    = f"{patient_id}_{joint}.png"
        img_path    = self.image_dir / filename

        # ── Load image ────────────────────────────────────────────────────────
        try:
            image = Image.open(img_path).convert("L")    # grayscale
        except FileNotFoundError:
            # Fallback: return a black image so training doesn't crash;
            # you can instead raise here to catch data issues early.
            image = Image.fromarray(np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8))

        image      = self.transform(image)
        kl_label   = torch.tensor(int(row["v00_KL"]),      dtype=torch.long)
        jt_label   = torch.tensor(int(row["joint_type_id"]), dtype=torch.long)

        return image, kl_label, jt_label


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Generator
# ─────────────────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    z (100) + emb_kl (16) + emb_joint (8)  →  1×180×180

    Spatial progression  (see module docstring for math):
      5 → 10 → 20 → 40 → 80  via ConvTranspose2d (kernel=4, stride=2, pad=1)
      80 → 180                via bilinear Upsample
      180 → 180               via refinement Conv2d + Tanh
    """

    def __init__(self):
        super().__init__()
        self.emb_kl    = nn.Embedding(N_KL,    EMB_KL)
        self.emb_joint = nn.Embedding(N_JOINT, EMB_JOINT)

        input_dim = LATENT_DIM + EMB_KL + EMB_JOINT  # 100+16+8 = 124

        # ── Project to 512×5×5 feature map ───────────────────────────────────
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 512 * 5 * 5),
            nn.BatchNorm1d(512 * 5 * 5),
            nn.ReLU(True),
        )

        # ── Upsampling blocks: each doubles spatial size ──────────────────────
        # kernel=4, stride=2, padding=1  →  H_out = 2*H_in  (exact)
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch,
                                   kernel_size=4, stride=2, padding=1,
                                   bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(True),
            )

        self.up1 = up_block(512, 256)   #   5 → 10
        self.up2 = up_block(256, 128)   #  10 → 20
        self.up3 = up_block(128,  64)   #  20 → 40
        self.up4 = up_block( 64,  32)   #  40 → 80

        # ── Stretch 80 → 180 (factor 2.25) then refine ───────────────────────
        self.to_img = nn.Sequential(
            nn.Upsample(size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False),
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, kl_label, jt_label):
        e_kl    = self.emb_kl(kl_label)          # (B, 16)
        e_joint = self.emb_joint(jt_label)        # (B,  8)
        x       = torch.cat([z, e_kl, e_joint], dim=1)  # (B, 124)

        x = self.proj(x)                          # (B, 512*25)
        x = x.view(-1, 512, 5, 5)                # (B, 512, 5, 5)

        x = self.up1(x)                           # (B, 256, 10, 10)
        x = self.up2(x)                           # (B, 128, 20, 20)
        x = self.up3(x)                           # (B,  64, 40, 40)
        x = self.up4(x)                           # (B,  32, 80, 80)

        return self.to_img(x)                     # (B,   1,180,180)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Discriminator
# ─────────────────────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    (1×180×180 image) + label embeddings → real/fake probability

    Label conditioning strategy:
      KL and joint embeddings are projected to 1-channel 180×180 maps and
      concatenated with the image as extra input channels (total = 3 ch).
      This lets the discriminator see spatial context alongside labels.

    Spatial progression (Conv2d stride=2, kernel=4, padding=1):
      180 → 90 → 45 → 22 → 11 → 5 → GlobalAvgPool → sigmoid
    """

    def __init__(self):
        super().__init__()
        self.emb_kl    = nn.Embedding(N_KL,    EMB_KL)
        self.emb_joint = nn.Embedding(N_JOINT, EMB_JOINT)

        # Project label embeddings to a single spatial channel
        self.label_proj = nn.Linear(EMB_KL + EMB_JOINT, IMG_SIZE * IMG_SIZE)

        # ── Convolutional backbone ────────────────────────────────────────────
        # Input: image (1 ch) + label map (1 ch) = 2 channels
        def down_block(in_ch, out_ch, bn=True):
            layers = [
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=4, stride=2, padding=1, bias=False),
            ]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.conv_blocks = nn.Sequential(
            down_block(  2,  64, bn=False),   # 180 →  90
            down_block( 64, 128),              #  90 →  45
            down_block(128, 256),              #  45 →  22  (floor((45-1)/2+1)=23 → actually 22 with pad)
            down_block(256, 512),              #  22 →  11
            down_block(512, 512),              #  11 →   5  (floor((11-1)/2+1)=6 → 5 with pad)
        )

        # ── Global average pool then classify ─────────────────────────────────
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (B, 512, 1, 1)
            nn.Flatten(),                      # (B, 512)
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, kl_label, jt_label):
        B = img.size(0)

        e_kl    = self.emb_kl(kl_label)                          # (B, 16)
        e_joint = self.emb_joint(jt_label)                       # (B,  8)
        label   = torch.cat([e_kl, e_joint], dim=1)              # (B, 24)
        label   = self.label_proj(label)                         # (B, 180*180)
        label   = label.view(B, 1, IMG_SIZE, IMG_SIZE)           # (B, 1,180,180)

        x = torch.cat([img, label], dim=1)                       # (B, 2,180,180)
        x = self.conv_blocks(x)
        return self.classifier(x)                                 # (B, 1)


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Weight initialisation (DCGAN standard)
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
# 7.  Visualisation helper
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def save_sample_grid(generator, epoch, n_per_kl=4):
    """
    Generates a grid of images: one row per KL score (0–4),
    n_per_kl samples per row, joint_type fixed to DIP (0).
    Saves as PNG and displays inline in Jupyter.
    """
    generator.eval()
    all_imgs = []

    for kl in range(N_KL):
        z        = torch.randn(n_per_kl, LATENT_DIM, device=DEVICE)
        kl_t     = torch.full((n_per_kl,), kl,  dtype=torch.long, device=DEVICE)
        jt_t     = torch.zeros(n_per_kl,        dtype=torch.long, device=DEVICE)  # DIP
        imgs     = generator(z, kl_t, jt_t)     # [-1,1]
        all_imgs.append(imgs.cpu())

    all_imgs = torch.cat(all_imgs, dim=0)        # (N_KL*n_per_kl, 1, 180, 180)

    grid = vutils.make_grid(
        all_imgs, nrow=n_per_kl, normalize=True, value_range=(-1, 1), padding=4
    )

    save_path = OUTPUT_DIR / f"samples_epoch_{epoch:04d}.png"
    vutils.save_image(grid, save_path)

    print(f"  [Saved sample grid → {save_path}]")  # just print not show

    generator.train()
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Training loop
# ─────────────────────────────────────────────────────────────────────────────
def make_weighted_sampler(dataset):
    # Count samples per (KL, joint_type) combination
    labels = [
        (int(row["v00_KL"]), int(row["joint_type_id"]))
        for _, row in dataset.df.iterrows()
    ]

    # Count how many of each combination exist
    from collections import Counter
    counts = Counter(labels)

    # Assign each sample a weight = 1 / count of its class
    # Rare classes get high weight, common classes get low weight
    weights = [1.0 / counts[l] for l in labels]
    weights = torch.tensor(weights, dtype=torch.float)

    return WeightedRandomSampler(
        weights,
        num_samples = len(weights),
        replacement = True,        # must be True for weighted sampling
    )

def train_cgan():
    # ── Data ──────────────────────────────────────────────────────────────────
    dataset    = FingerJointDataset(CSV_PATH, IMAGE_DIR)
    sampler    = make_weighted_sampler(dataset)
    dataloader = DataLoader(
        dataset, batch_size = BATCH_SIZE,
        sampler = sampler,
        num_workers = 4,
        pin_memory = True,
        drop_last = True,
    )
    print(f"Dataset size : {len(dataset):,} images")
    print(f"Batches/epoch: {len(dataloader)}")

    # ── Models ────────────────────────────────────────────────────────────────
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)
    G.apply(weights_init)
    D.apply(weights_init)
    print(f"\nGenerator params    : {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    # ── Loss & Optimisers ─────────────────────────────────────────────────────
    criterion = nn.BCELoss()
    opt_G     = optim.Adam(G.parameters(), lr=LR * 0.5, betas=(BETA1, 0.999))
    opt_D     = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))

    # Fixed noise for consistent visualisation across epochs
    fixed_z  = torch.randn(N_KL * 4, LATENT_DIM, device=DEVICE)
    fixed_kl = torch.repeat_interleave(torch.arange(N_KL, device=DEVICE), 4)
    fixed_jt = torch.zeros(N_KL * 4, dtype=torch.long, device=DEVICE)

    # ── Tracking ──────────────────────────────────────────────────────────────
    history = {"d_loss": [], "g_loss": [], "d_real": [], "d_fake": []}

    # ── Main loop ─────────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_d, epoch_g, epoch_dr, epoch_df = [], [], [], []

        for real_imgs, kl_labels, jt_labels in dataloader:
            B = real_imgs.size(0)
            real_imgs  = real_imgs.to(DEVICE)
            kl_labels  = kl_labels.to(DEVICE)
            jt_labels  = jt_labels.to(DEVICE)

            real_label = torch.full((B, 1), 0.9, device=DEVICE)
            fake_label = torch.zeros(B, 1, device=DEVICE)

            # ──────────────────────────────────────────────────────────────────
            # (A) Update Discriminator
            # ──────────────────────────────────────────────────────────────────
            opt_D.zero_grad()

            # Real images
            out_real = D(real_imgs, kl_labels, jt_labels)
            d_loss_real = criterion(out_real, real_label)

            # Fake images
            z        = torch.randn(B, LATENT_DIM, device=DEVICE)
            fake_imgs = G(z, kl_labels, jt_labels).detach()
            out_fake  = D(fake_imgs, kl_labels, jt_labels)
            d_loss_fake = criterion(out_fake, fake_label)

            d_loss = (d_loss_real + d_loss_fake) * 0.5
            d_loss.backward()
            opt_D.step()

            # ──────────────────────────────────────────────────────────────────
            # (B) Update Generator
            # ──────────────────────────────────────────────────────────────────
            for _ in range(2):
                opt_G.zero_grad()
                z         = torch.randn(B, LATENT_DIM, device=DEVICE)
                fake_imgs = G(z, kl_labels, jt_labels)
                out_fake  = D(fake_imgs, kl_labels, jt_labels)
                g_loss    = criterion(out_fake, real_label)   # fool D
                g_loss.backward()
                opt_G.step()

            # ── Book-keeping ──────────────────────────────────────────────────
            epoch_d.append(d_loss.item())
            epoch_g.append(g_loss.item())
            epoch_dr.append(out_real.mean().item())
            epoch_df.append(out_fake.mean().item())

        # ── Epoch summary ─────────────────────────────────────────────────────
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

        # ── Visualise ─────────────────────────────────────────────────────────
        if epoch % SAVE_EVERY == 0 or epoch == 1:
            save_sample_grid(G, epoch)
            # Save checkpoints
            torch.save(G.state_dict(), CHECKPOINT_DIR / f"G_epoch_{epoch:04d}.pth")
            torch.save(D.state_dict(), CHECKPOINT_DIR / f"D_epoch_{epoch:04d}.pth")

    # ── Plot training curves ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].plot(history["d_loss"], label="D loss")
    axes[0].plot(history["g_loss"], label="G loss")
    axes[0].set_title("Generator & Discriminator Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["d_real"], label="D(real)")
    axes[1].plot(history["d_fake"], label="D(fake)")
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_title("Average Discriminator Outputs")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_curves.png", dpi=150)
    plt.show()

    return G, D, history


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Generation utility  (post-training)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_synthetic_images(generator, n_per_class=200,
                               kl_scores=None, joint_types=None):
    """
    Generate `n_per_class` images for every (KL score, joint type) combination.

    Args:
        generator   : trained Generator
        n_per_class : images per (KL, joint_type) pair
        kl_scores   : list of KL values to generate (default: 0-4)
        joint_types : list of joint type ids (default: 0,1,2)

    Returns:
        dict keyed by (kl, jt) → list of PIL images
    """
    generator.eval()
    kl_scores   = kl_scores   or list(range(N_KL))
    joint_types = joint_types or list(range(N_JOINT))
    inv_jt = {v: k for k, v in JOINT_TYPE_MAP.items()}
    results = {}

    for kl in kl_scores:
        for jt in joint_types:
            imgs_out = []
            remaining = n_per_class
            while remaining > 0:
                bs = min(BATCH_SIZE, remaining)
                z     = torch.randn(bs, LATENT_DIM, device=DEVICE)
                kl_t  = torch.full((bs,), kl, dtype=torch.long, device=DEVICE)
                jt_t  = torch.full((bs,), jt, dtype=torch.long, device=DEVICE)
                out   = generator(z, kl_t, jt_t).cpu()         # [-1,1]
                out   = (out * 0.5 + 0.5).clamp(0, 1)          # [0,1]
                out   = (out * 255).byte().squeeze(1).numpy()   # (bs,180,180) uint8
                imgs_out.extend([Image.fromarray(o) for o in out])
                remaining -= bs

            results[(kl, jt)] = imgs_out
            print(f"  KL={kl}  {inv_jt[jt]:<3}  → {len(imgs_out)} images generated")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    G, D, history = train_cgan()

    # Example: generate 200 synthetic images per (KL, joint_type) pair
    # synthetic = generate_synthetic_images(G, n_per_class=200)