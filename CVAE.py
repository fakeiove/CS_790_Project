# ============================================================
# dual_condition_cvae_jointtype_kl_final_fixed_id.py
#
# One-model dual-condition CVAE
# Conditions:
#   1) joint_type: DIP / PIP / MCP
#   2) KL grade : 0 / 1 / 2 / 3 / 4
#
# Matched to real filename rule:
#   9000099_dip2.png
#   9000099_pip3.png
#   9000099_mcp4.png
#
# IMPORTANT FIX:
#   Excel ID column is fixed to "id"
#   NOT "duryeaid"
# ============================================================

import os
import re
import math
import json
import random
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, utils


# ============================================================
# 1. Reproducibility
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)


# ============================================================
# 2. Config
# ============================================================
class CFG:
    IMAGE_DIR = "./Finger Joints"
    EXCEL_PATH = "./hand.xlsx"
    OUTPUT_DIR = "./output_dual_condition_cvae_final"

    IMG_SIZE = 128
    BATCH_SIZE = 16
    EPOCHS = 120
    LR = 2e-4
    WEIGHT_DECAY = 1e-5

    LATENT_DIM = 128
    BASE_CH = 32

    JOINT_EMB_DIM = 8
    KL_EMB_DIM = 12

    VAL_RATIO = 0.15

    USE_SSIM_LOSS = True
    LAMBDA_SSIM = 0.15

    KL_BETA_START = 0.0
    KL_BETA_END = 1e-3
    KL_WARMUP_EPOCHS = 20

    KL_CLASS_WEIGHTS = {
        0: 1.0,
        1: 1.2,
        2: 1.4,
        3: 5.0,
        4: 7.0,
    }

    PAIR_EXTRA_WEIGHT = {
        ("DIP", 3): 1.4,
        ("DIP", 4): 1.8,
        ("PIP", 3): 1.4,
        ("PIP", 4): 1.8,
        ("MCP", 3): 1.4,
        ("MCP", 4): 1.8,
    }

    GENERATE_TARGETS = [
        ("DIP", 3),
        ("DIP", 4),
        ("PIP", 3),
        ("PIP", 4),
        ("MCP", 3),
        ("MCP", 4),
    ]

    N_SAMPLES_PER_TARGET = 50
    REAL_LATENT_GUIDED = True
    GUIDED_NOISE_STD = 0.12

    SAVE_EVERY = 5
    NUM_WORKERS = 0

cfg = CFG()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(cfg.OUTPUT_DIR, "samples"), exist_ok=True)
os.makedirs(os.path.join(cfg.OUTPUT_DIR, "generated"), exist_ok=True)
os.makedirs(os.path.join(cfg.OUTPUT_DIR, "checkpoints"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEVICE)
if torch.cuda.is_available():
    print("[GPU]", torch.cuda.get_device_name(0))


# ============================================================
# 3. Constants / Mappings
# ============================================================
JOINT_TYPES = ["DIP", "PIP", "MCP"]
JOINT_TYPE_TO_ID = {"DIP": 0, "PIP": 1, "MCP": 2}
ID_TO_JOINT_TYPE = {v: k for k, v in JOINT_TYPE_TO_ID.items()}

VALID_JOINT_CODES = [
    "DIP2", "DIP3", "DIP4", "DIP5",
    "PIP2", "PIP3", "PIP4", "PIP5",
    "MCP2", "MCP3", "MCP4", "MCP5",
]

NUM_JOINT_TYPES = 3
NUM_KL_CLASSES = 5


# ============================================================
# 4. Helpers
# ============================================================
def normalize_id(x):
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def extract_image_id(filename):
    # 9000099_dip2.png -> 9000099
    base = os.path.splitext(filename)[0]
    return base.split("_")[0]

def extract_joint_from_filename(filename):
    # 9000099_dip2.png -> DIP2
    base = os.path.splitext(filename)[0].lower()
    m = re.search(r'_(dip|pip|mcp)([2-5])$', base)
    if m is None:
        return None
    return f"{m.group(1).upper()}{m.group(2)}"

def get_joint_type_from_joint_code(joint_code):
    if joint_code.startswith("DIP"):
        return "DIP"
    if joint_code.startswith("PIP"):
        return "PIP"
    if joint_code.startswith("MCP"):
        return "MCP"
    return None

def save_image_grid(tensor_batch, save_path, nrow=8):
    grid = utils.make_grid(tensor_batch, nrow=nrow, padding=2, normalize=False)
    utils.save_image(grid, save_path)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)

def kl_beta_schedule(epoch, warmup_epochs, beta_start, beta_end):
    if epoch <= 0:
        return beta_start
    if epoch >= warmup_epochs:
        return beta_end
    alpha = epoch / float(warmup_epochs)
    return beta_start + alpha * (beta_end - beta_start)


# ============================================================
# 5. SSIM
# ============================================================
def gaussian_window(window_size=11, sigma=1.5, channels=1, device="cpu"):
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    window_2d = window_1d @ window_1d.t()
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)
    return window_2d.expand(channels, 1, window_size, window_size).contiguous()

def ssim_torch(img1, img2, window_size=11, sigma=1.5, data_range=1.0, eps=1e-8):
    channels = img1.size(1)
    window = gaussian_window(window_size, sigma, channels, img1.device)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channels)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channels) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + eps)
    return ssim_map.mean()


# ============================================================
# 6. Load metadata and build label map
# ============================================================
def build_joint_kl_column_map(df):
    col_map = {}

    for joint in VALID_JOINT_CODES:
        candidates = [
            f"v00{joint}_KL",
            f"{joint}_KL",
            f"{joint}KL",
            f"KL_{joint}",
            f"kl_{joint}",
            f"{joint.lower()}_kl",
        ]

        found = None
        for c in candidates:
            if c in df.columns:
                found = c
                break

        if found is None:
            for c in df.columns:
                c_norm = c.replace(" ", "").replace("-", "").lower()
                possible = [
                    f"v00{joint.lower()}_kl",
                    f"{joint.lower()}_kl",
                    f"{joint.lower()}kl",
                ]
                if c_norm in possible:
                    found = c
                    break

        if found is not None:
            col_map[joint] = found
        else:
            print(f"[Warning] Could not find KL column for {joint}")

    return col_map

def load_metadata():
    df = pd.read_excel(cfg.EXCEL_PATH)

    # ===== FIXED HERE =====
    id_col = "id"
    if id_col not in df.columns:
        raise ValueError(f"Excel does not contain required id column '{id_col}'. Available columns: {list(df.columns)}")
    # ======================

    joint_col_map = build_joint_kl_column_map(df)

    missing = [j for j in VALID_JOINT_CODES if j not in joint_col_map]
    if missing:
        raise ValueError(f"Missing KL columns for joints: {missing}")

    metadata = {}
    for _, row in df.iterrows():
        pid = normalize_id(row[id_col])
        if pid is None:
            continue

        metadata[pid] = {}
        for joint_code, col in joint_col_map.items():
            val = row[col]
            if pd.notna(val):
                try:
                    kl = int(val)
                    if 0 <= kl <= 4:
                        metadata[pid][joint_code] = kl
                except Exception:
                    pass

    return metadata


# ============================================================
# 7. Build matched items from image filenames
# ============================================================
def build_items(metadata):
    items = []
    skipped = []

    for fname in os.listdir(cfg.IMAGE_DIR):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue

        pid = extract_image_id(fname)
        joint_code = extract_joint_from_filename(fname)

        if joint_code is None:
            skipped.append((fname, "joint_not_recognized"))
            continue

        if pid not in metadata:
            skipped.append((fname, "id_not_found_in_excel"))
            continue

        if joint_code not in metadata[pid]:
            skipped.append((fname, f"missing_kl_for_{joint_code}"))
            continue

        kl = metadata[pid][joint_code]
        joint_type = get_joint_type_from_joint_code(joint_code)

        items.append({
            "img_path": os.path.join(cfg.IMAGE_DIR, fname),
            "id": pid,
            "joint_code": joint_code,
            "joint_type": joint_type,
            "joint_type_id": JOINT_TYPE_TO_ID[joint_type],
            "kl": int(kl),
        })

    print(f"[Dataset] usable images: {len(items)}")
    print(f"[Dataset] skipped images: {len(skipped)}")
    if skipped:
        print("[Dataset] first few skipped files:")
        for x in skipped[:10]:
            print(" ", x)

    dist = defaultdict(int)
    for item in items:
        dist[(item["joint_type"], item["kl"])] += 1

    print("\n[Distribution] joint_type x KL")
    for jt in JOINT_TYPES:
        row = []
        for kl in range(5):
            row.append(f"KL{kl}={dist[(jt, kl)]}")
        print(f"{jt}: " + ", ".join(row))

    return items, skipped


# ============================================================
# 8. Dataset
# ============================================================
class JointKLDataset(Dataset):
    def __init__(self, items, img_size=128):
        self.items = items
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["img_path"]).convert("L")
        img = self.transform(img)

        meta = {
            "id": item["id"],
            "joint_code": item["joint_code"],
            "joint_type": item["joint_type"],
            "img_path": item["img_path"],
        }

        return (
            img,
            torch.tensor(item["joint_type_id"], dtype=torch.long),
            torch.tensor(item["kl"], dtype=torch.long),
            meta
        )


# ============================================================
# 9. Model blocks
# ============================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, use_bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class DeconvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=4, s=2, p=1, use_bn=True):
        super().__init__()
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        attn = torch.softmax(torch.bmm(q, k), dim=-1)
        v = self.value(x).view(b, c, h * w)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        return self.gamma * out + x


# ============================================================
# 10. Dual-condition CVAE
# ============================================================
class DualConditionCVAE(nn.Module):
    def __init__(
        self,
        img_size=128,
        latent_dim=128,
        num_joint_types=3,
        num_kl_classes=5,
        joint_emb_dim=8,
        kl_emb_dim=12,
        base_ch=32
    ):
        super().__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.base_ch = base_ch

        self.joint_emb = nn.Embedding(num_joint_types, joint_emb_dim)
        self.kl_emb = nn.Embedding(num_kl_classes, kl_emb_dim)
        cond_dim = joint_emb_dim + kl_emb_dim

        self.enc1 = ConvBlock(1, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.attn = SelfAttention2d(base_ch * 8)

        self.feature_dim = (base_ch * 8) * 8 * 8

        self.fc_enc = nn.Linear(self.feature_dim + cond_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + cond_dim, self.feature_dim)

        self.dec1 = DeconvBlock(base_ch * 8, base_ch * 4)
        self.dec2 = DeconvBlock(base_ch * 4, base_ch * 2)
        self.dec3 = DeconvBlock(base_ch * 2, base_ch)
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(base_ch, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def get_cond(self, joint_type_id, kl):
        j = self.joint_emb(joint_type_id)
        k = self.kl_emb(kl)
        return torch.cat([j, k], dim=1)

    def encode_features(self, x):
        h = self.enc1(x)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc4(h)
        h = self.attn(h)
        return h

    def encode(self, x, joint_type_id, kl):
        h = self.encode_features(x)
        h = h.view(x.size(0), -1)
        cond = self.get_cond(joint_type_id, kl)
        h = torch.cat([h, cond], dim=1)
        h = F.relu(self.fc_enc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * temperature
        return mu + eps * std

    def decode(self, z, joint_type_id, kl):
        cond = self.get_cond(joint_type_id, kl)
        h = torch.cat([z, cond], dim=1)
        h = self.fc_dec(h)
        h = h.view(-1, self.base_ch * 8, 8, 8)
        h = self.attn(h)
        h = self.dec1(h)
        h = self.dec2(h)
        h = self.dec3(h)
        x_recon = self.dec4(h)
        return x_recon

    def forward(self, x, joint_type_id, kl):
        mu, logvar = self.encode(x, joint_type_id, kl)
        z = self.reparameterize(mu, logvar, temperature=1.0)
        recon = self.decode(z, joint_type_id, kl)
        return recon, mu, logvar


# ============================================================
# 11. Loss
# ============================================================
def cvae_loss(recon_x, x, mu, logvar, beta, use_ssim_loss=False, lambda_ssim=0.15):
    l1_loss = F.l1_loss(recon_x, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    ssim_value = ssim_torch(recon_x, x)
    ssim_loss = 1.0 - ssim_value

    total = l1_loss + beta * kl_loss
    if use_ssim_loss:
        total = total + lambda_ssim * ssim_loss

    return total, l1_loss, kl_loss, ssim_value, ssim_loss


# ============================================================
# 12. Build weighted sampler
# ============================================================
def build_weighted_sampler(train_items):
    weights = []
    for item in train_items:
        kl = item["kl"]
        jt = item["joint_type"]

        w = cfg.KL_CLASS_WEIGHTS.get(kl, 1.0)
        w *= cfg.PAIR_EXTRA_WEIGHT.get((jt, kl), 1.0)
        weights.append(w)

    weights = torch.tensor(weights, dtype=torch.double)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_items),
        replacement=True
    )
    return sampler, weights.tolist()


# ============================================================
# 13. One epoch runner
# ============================================================
def run_epoch(model, loader, beta, optimizer=None, train_mode=True):
    if train_mode:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    total_ssim = 0.0
    total_ssim_loss = 0.0

    for imgs, joint_type_ids, kls, _ in loader:
        imgs = imgs.to(DEVICE)
        joint_type_ids = joint_type_ids.to(DEVICE)
        kls = kls.to(DEVICE)

        if train_mode:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train_mode):
            recon, mu, logvar = model(imgs, joint_type_ids, kls)
            loss, l1_loss, kl_loss, ssim_value, ssim_loss = cvae_loss(
                recon, imgs, mu, logvar,
                beta=beta,
                use_ssim_loss=cfg.USE_SSIM_LOSS,
                lambda_ssim=cfg.LAMBDA_SSIM
            )

            if train_mode:
                loss.backward()
                optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_l1 += l1_loss.item() * bs
        total_kl += kl_loss.item() * bs
        total_ssim += ssim_value.item() * bs
        total_ssim_loss += ssim_loss.item() * bs

    n = len(loader.dataset)
    return (
        total_loss / n,
        total_l1 / n,
        total_kl / n,
        total_ssim / n,
        total_ssim_loss / n
    )


# ============================================================
# 14. Save recon preview
# ============================================================
@torch.no_grad()
def save_recon_examples(model, loader, out_path, max_samples=8):
    model.eval()

    imgs_all = []
    recon_all = []

    for imgs, joint_type_ids, kls, _ in loader:
        imgs = imgs.to(DEVICE)
        joint_type_ids = joint_type_ids.to(DEVICE)
        kls = kls.to(DEVICE)

        recon, _, _ = model(imgs, joint_type_ids, kls)
        imgs_all.append(imgs.cpu())
        recon_all.append(recon.cpu())

        if sum(x.size(0) for x in imgs_all) >= max_samples:
            break

    imgs_all = torch.cat(imgs_all, dim=0)[:max_samples]
    recon_all = torch.cat(recon_all, dim=0)[:max_samples]
    combined = torch.cat([imgs_all, recon_all], dim=0)

    save_image_grid(combined, out_path, nrow=max_samples)
    print(f"[Saved] {out_path}")


# ============================================================
# 15. Build data
# ============================================================
metadata = load_metadata()
all_items, skipped = build_items(metadata)

if len(all_items) == 0:
    raise RuntimeError("No usable items found. Please check filenames and hand.xlsx.")

dataset = JointKLDataset(all_items, img_size=cfg.IMG_SIZE)

val_size = int(len(dataset) * cfg.VAL_RATIO)
train_size = len(dataset) - val_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_items = [all_items[i] for i in train_dataset.indices]
val_items = [all_items[i] for i in val_dataset.indices]

sampler, sampling_weights = build_weighted_sampler(train_items)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    sampler=sampler,
    num_workers=cfg.NUM_WORKERS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS
)


# ============================================================
# 16. Model / optimizer
# ============================================================
model = DualConditionCVAE(
    img_size=cfg.IMG_SIZE,
    latent_dim=cfg.LATENT_DIM,
    num_joint_types=NUM_JOINT_TYPES,
    num_kl_classes=NUM_KL_CLASSES,
    joint_emb_dim=cfg.JOINT_EMB_DIM,
    kl_emb_dim=cfg.KL_EMB_DIM,
    base_ch=cfg.BASE_CH
).to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.LR,
    weight_decay=cfg.WEIGHT_DECAY
)


# ============================================================
# 17. Training
# ============================================================
best_val_loss = float("inf")
best_model_path = os.path.join(cfg.OUTPUT_DIR, "checkpoints", "best_model.pth")
history = []

for epoch in range(1, cfg.EPOCHS + 1):
    beta_t = kl_beta_schedule(
        epoch=epoch,
        warmup_epochs=cfg.KL_WARMUP_EPOCHS,
        beta_start=cfg.KL_BETA_START,
        beta_end=cfg.KL_BETA_END
    )

    train_loss, train_l1, train_kl, train_ssim, train_ssim_loss = run_epoch(
        model, train_loader, beta=beta_t, optimizer=optimizer, train_mode=True
    )
    val_loss, val_l1, val_kl, val_ssim, val_ssim_loss = run_epoch(
        model, val_loader, beta=beta_t, optimizer=None, train_mode=False
    )

    history.append({
        "epoch": epoch,
        "beta_t": beta_t,
        "train_loss": train_loss,
        "train_l1": train_l1,
        "train_kl": train_kl,
        "train_ssim": train_ssim,
        "train_ssim_loss": train_ssim_loss,
        "val_loss": val_loss,
        "val_l1": val_l1,
        "val_kl": val_kl,
        "val_ssim": val_ssim,
        "val_ssim_loss": val_ssim_loss,
    })

    print(
        f"Epoch [{epoch:03d}/{cfg.EPOCHS}] | "
        f"beta={beta_t:.6f} | "
        f"Train {train_loss:.6f} (L1 {train_l1:.6f}, KL {train_kl:.6f}, SSIM {train_ssim:.4f}) | "
        f"Val {val_loss:.6f} (L1 {val_l1:.6f}, KL {val_kl:.6f}, SSIM {val_ssim:.4f})"
    )

    if epoch % cfg.SAVE_EVERY == 0 or epoch == 1:
        recon_out = os.path.join(cfg.OUTPUT_DIR, "samples", f"recon_epoch_{epoch:03d}.png")
        save_recon_examples(model, val_loader, recon_out, max_samples=8)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"  -> saved best model to {best_model_path}")

history_path = os.path.join(cfg.OUTPUT_DIR, "training_history.json")
save_json(history, history_path)

summary = {
    "num_total": len(all_items),
    "num_train": len(train_items),
    "num_val": len(val_items),
    "best_val_loss": best_val_loss,
    "best_model_path": best_model_path,
    "sampling_weights_preview": sampling_weights[:20],
}
save_json(summary, os.path.join(cfg.OUTPUT_DIR, "summary.json"))


# ============================================================
# 18. Reload best
# ============================================================
model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))
model.eval()


# ============================================================
# 19. Collect latent bank by (joint_type, kl)
# ============================================================
@torch.no_grad()
def collect_latent_bank(model, items):
    bank = defaultdict(list)
    meta_bank = defaultdict(list)

    infer_ds = JointKLDataset(items, img_size=cfg.IMG_SIZE)
    infer_loader = DataLoader(infer_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0)

    for imgs, joint_type_ids, kls, metas in infer_loader:
        imgs = imgs.to(DEVICE)
        joint_type_ids = joint_type_ids.to(DEVICE)
        kls = kls.to(DEVICE)

        mu, logvar = model.encode(imgs, joint_type_ids, kls)

        for i in range(imgs.size(0)):
            jt_id = int(joint_type_ids[i].item())
            kl = int(kls[i].item())
            key = (jt_id, kl)

            bank[key].append(mu[i].detach().cpu())

            meta_bank[key].append({
                "id": metas["id"][i],
                "joint_code": metas["joint_code"][i],
                "joint_type": metas["joint_type"][i],
                "img_path": metas["img_path"][i],
            })

    final_bank = {}
    for k, v in bank.items():
        final_bank[k] = torch.stack(v, dim=0) if len(v) > 0 else None

    return final_bank, meta_bank

latent_bank, latent_meta_bank = collect_latent_bank(model, all_items)


# ============================================================
# 20. Generation
# ============================================================
@torch.no_grad()
def generate_from_prior(model, joint_type_name, kl, n=16, temperature=1.0):
    jt_id = JOINT_TYPE_TO_ID[joint_type_name]
    z = torch.randn(n, cfg.LATENT_DIM, device=DEVICE) * temperature
    jt_tensor = torch.full((n,), jt_id, dtype=torch.long, device=DEVICE)
    kl_tensor = torch.full((n,), kl, dtype=torch.long, device=DEVICE)
    fake = model.decode(z, jt_tensor, kl_tensor)
    return fake

@torch.no_grad()
def generate_real_latent_guided(model, joint_type_name, kl, n=16, noise_std=0.12):
    jt_id = JOINT_TYPE_TO_ID[joint_type_name]
    key = (jt_id, kl)

    if key not in latent_bank or latent_bank[key] is None or len(latent_bank[key]) == 0:
        return generate_from_prior(model, joint_type_name, kl, n=n, temperature=1.0)

    bank = latent_bank[key].to(DEVICE)
    idx = torch.randint(0, bank.size(0), (n,), device=DEVICE)
    z_real = bank[idx]
    noise = torch.randn_like(z_real) * noise_std
    z = z_real + noise

    jt_tensor = torch.full((n,), jt_id, dtype=torch.long, device=DEVICE)
    kl_tensor = torch.full((n,), kl, dtype=torch.long, device=DEVICE)
    fake = model.decode(z, jt_tensor, kl_tensor)
    return fake

@torch.no_grad()
def save_generated_samples(model, targets, n_per_target=50):
    for joint_type_name, kl in targets:
        target_dir = os.path.join(cfg.OUTPUT_DIR, "generated", f"{joint_type_name}_KL{kl}")
        os.makedirs(target_dir, exist_ok=True)

        saved = 0
        batch_idx = 0
        batch_size = 16

        while saved < n_per_target:
            cur_n = min(batch_size, n_per_target - saved)

            if cfg.REAL_LATENT_GUIDED:
                fake = generate_real_latent_guided(
                    model=model,
                    joint_type_name=joint_type_name,
                    kl=kl,
                    n=cur_n,
                    noise_std=cfg.GUIDED_NOISE_STD
                )
            else:
                fake = generate_from_prior(
                    model=model,
                    joint_type_name=joint_type_name,
                    kl=kl,
                    n=cur_n,
                    temperature=1.0
                )

            fake = fake.cpu()

            for i in range(fake.size(0)):
                out_path = os.path.join(target_dir, f"{joint_type_name}_KL{kl}_{saved+i:04d}.png")
                utils.save_image(fake[i], out_path)

            sheet_path = os.path.join(target_dir, f"sheet_{batch_idx:03d}.png")
            save_image_grid(fake, sheet_path, nrow=min(4, fake.size(0)))

            saved += cur_n
            batch_idx += 1

        print(f"[Generated] {joint_type_name} KL{kl} -> {target_dir}")

save_generated_samples(
    model=model,
    targets=cfg.GENERATE_TARGETS,
    n_per_target=cfg.N_SAMPLES_PER_TARGET
)


# ============================================================
# 21. Latent interpolation
# ============================================================
@torch.no_grad()
def latent_interpolation(model, joint_type_name="DIP", kl=4, steps=8, out_path="interp.png"):
    jt_id = JOINT_TYPE_TO_ID[joint_type_name]
    key = (jt_id, kl)

    if key not in latent_bank or latent_bank[key] is None or len(latent_bank[key]) < 2:
        print(f"[Warning] Not enough latent samples for {joint_type_name} KL{kl}")
        return

    bank = latent_bank[key].to(DEVICE)
    idx = torch.randperm(bank.size(0), device=DEVICE)[:2]
    z1, z2 = bank[idx[0]], bank[idx[1]]

    z_list = []
    for a in torch.linspace(0, 1, steps, device=DEVICE):
        z = (1 - a) * z1 + a * z2
        z_list.append(z.unsqueeze(0))

    z_all = torch.cat(z_list, dim=0)
    jt_tensor = torch.full((steps,), jt_id, dtype=torch.long, device=DEVICE)
    kl_tensor = torch.full((steps,), kl, dtype=torch.long, device=DEVICE)

    imgs = model.decode(z_all, jt_tensor, kl_tensor).cpu()
    save_image_grid(imgs, out_path, nrow=steps)
    print(f"[Saved interpolation] {out_path}")

latent_interpolation(
    model=model,
    joint_type_name="DIP",
    kl=4,
    steps=8,
    out_path=os.path.join(cfg.OUTPUT_DIR, "latent_interp_DIP_KL4.png")
)

print("\n[All Done] Dual-condition CVAE training and generation completed.")