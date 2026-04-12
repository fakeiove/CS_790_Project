# ============================================================
# sharp_unet_dual_condition_cvae_dip_only_dp.py
#
# 2-GPU DataParallel final runnable version
#
# Sharp U-Net style dual-condition CVAE
# Conditions:
#   1) joint_type: DIP / PIP / MCP
#   2) KL grade : 0 / 1 / 2 / 3 / 4
#
# Main upgrades:
#   - U-Net style decoder with skip connections
#   - DIP KL3-4 only generation
#   - training set = original + limited augmentation
#   - augmentation <= 50% of original train size
#   - augment types: flip / rotate -3 / rotate +3
#   - prototype skip guided generation
#   - 2-GPU DataParallel support
#
# Matched to filename rule:
#   9000099_dip2.png
#   9000099_pip3.png
#   9000099_mcp4.png
#
# Excel ID column fixed to:
#   id
# ============================================================

import os
import re
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
    OUTPUT_DIR = "./output_sharp_unet_cvae_dip_only_dp"

    IMG_SIZE = 128
    BATCH_SIZE = 128
    EPOCHS = 1000
    LR = 2e-4
    WEIGHT_DECAY = 1e-5

    LATENT_DIM = 128
    BASE_CH = 48

    JOINT_EMB_DIM = 8
    KL_EMB_DIM = 12

    VAL_RATIO = 0.15

    USE_SSIM_LOSS = True
    LAMBDA_SSIM = 0.10

    KL_BETA_START = 0.0
    KL_BETA_END = 1e-3
    KL_WARMUP_EPOCHS = 40

    KL_CLASS_WEIGHTS = {
        0: 1.0,
        1: 1.2,
        2: 1.4,
        3: 5.0,
        4: 7.0,
    }

    PAIR_EXTRA_WEIGHT = {
        ("DIP", 3): 1.6,
        ("DIP", 4): 2.0,
        ("PIP", 3): 1.2,
        ("PIP", 4): 1.4,
        ("MCP", 3): 1.2,
        ("MCP", 4): 1.4,
    }

    GENERATE_TARGETS = [
        ("DIP", 3),
        ("DIP", 4),
    ]

    N_SAMPLES_PER_TARGET = 1500
    REAL_LATENT_GUIDED = True
    GUIDED_NOISE_STD = 0.08

    SAVE_EVERY = 20
    NUM_WORKERS = 4

    AUG_MAX_RATIO = 0.5
    AUG_TYPES = ["flip", "rot_left", "rot_right"]

cfg = CFG()

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(cfg.OUTPUT_DIR, "samples"), exist_ok=True)
os.makedirs(os.path.join(cfg.OUTPUT_DIR, "generated"), exist_ok=True)
os.makedirs(os.path.join(cfg.OUTPUT_DIR, "checkpoints"), exist_ok=True)


# ============================================================
# 3. Device / GPU check
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEVICE)

if torch.cuda.is_available():
    n_gpu = torch.cuda.device_count()
    print("[CUDA available] True")
    print(f"[Visible GPU count] {n_gpu}")
    for i in range(n_gpu):
        print(f"[GPU {i}] {torch.cuda.get_device_name(i)}")
else:
    print("[CUDA available] False")


# ============================================================
# 4. Constants / Mappings
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
# 5. Helpers
# ============================================================
def normalize_id(x):
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        return str(x).strip()

def extract_image_id(filename):
    base = os.path.splitext(filename)[0]
    return base.split("_")[0]

def extract_joint_from_filename(filename):
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
# 6. SSIM
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
# 7. Load metadata and build label map
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

    id_col = "id"
    if id_col not in df.columns:
        raise ValueError(
            f"Excel does not contain required id column '{id_col}'. "
            f"Available columns: {list(df.columns)}"
        )

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
# 8. Build matched items from image filenames
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
# 9. Dataset
# ============================================================
class JointKLDataset(Dataset):
    def __init__(self, items, img_size=128):
        self.items = items

        self.base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
        ])

        self.to_tensor = transforms.ToTensor()

        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)

        self.rotate_left = transforms.RandomRotation(
            degrees=(-3, -3),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )

        self.rotate_right = transforms.RandomRotation(
            degrees=(3, 3),
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(item["img_path"]).convert("L")
        img = self.base_transform(img)

        aug_type = item.get("aug_type", "none")

        if aug_type == "flip":
            img = self.flip_transform(img)
        elif aug_type == "rot_left":
            img = self.rotate_left(img)
        elif aug_type == "rot_right":
            img = self.rotate_right(img)

        img = self.to_tensor(img)

        meta = {
            "id": item["id"],
            "joint_code": item["joint_code"],
            "joint_type": item["joint_type"],
            "img_path": item["img_path"],
            "aug_type": aug_type,
        }

        return (
            img,
            torch.tensor(item["joint_type_id"], dtype=torch.long),
            torch.tensor(item["kl"], dtype=torch.long),
            meta
        )


# ============================================================
# 10. Model blocks
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

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

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
# 11. U-Net Style Sharp Dual-condition CVAE
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
        base_ch=48
    ):
        super().__init__()

        self.img_size = img_size
        self.latent_dim = latent_dim
        self.base_ch = base_ch

        self.joint_emb = nn.Embedding(num_joint_types, joint_emb_dim)
        self.kl_emb = nn.Embedding(num_kl_classes, kl_emb_dim)
        self.cond_dim = joint_emb_dim + kl_emb_dim

        self.enc1 = ConvBlock(1, base_ch, s=2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, s=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, s=2)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, s=2)

        self.enc_refine1 = DoubleConv(base_ch, base_ch)
        self.enc_refine2 = DoubleConv(base_ch * 2, base_ch * 2)
        self.enc_refine3 = DoubleConv(base_ch * 4, base_ch * 4)
        self.enc_refine4 = DoubleConv(base_ch * 8, base_ch * 8)

        self.attn = SelfAttention2d(base_ch * 8)

        self.feature_dim = (base_ch * 8) * 8 * 8

        self.fc_enc = nn.Linear(self.feature_dim + self.cond_dim, 512)
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + self.cond_dim, self.feature_dim)

        self.dec_bottleneck = DoubleConv(base_ch * 8, base_ch * 8)

        self.up1 = UpBlock(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up2 = UpBlock(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up3 = UpBlock(base_ch * 2, base_ch, base_ch)

        self.up4 = nn.ConvTranspose2d(base_ch, base_ch // 2, kernel_size=4, stride=2, padding=1)
        self.final_refine = DoubleConv(base_ch // 2, base_ch // 2)
        self.final_out = nn.Sequential(
            nn.Conv2d(base_ch // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def get_cond(self, joint_type_id, kl):
        j = self.joint_emb(joint_type_id)
        k = self.kl_emb(kl)
        return torch.cat([j, k], dim=1)

    def encode_features(self, x):
        s1 = self.enc1(x)
        s1 = self.enc_refine1(s1)

        s2 = self.enc2(s1)
        s2 = self.enc_refine2(s2)

        s3 = self.enc3(s2)
        s3 = self.enc_refine3(s3)

        s4 = self.enc4(s3)
        s4 = self.enc_refine4(s4)
        s4 = self.attn(s4)

        return s1, s2, s3, s4

    def encode(self, x, joint_type_id, kl):
        s1, s2, s3, s4 = self.encode_features(x)
        h = s4.view(x.size(0), -1)
        cond = self.get_cond(joint_type_id, kl)
        h = torch.cat([h, cond], dim=1)
        h = F.relu(self.fc_enc(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, (s1, s2, s3)

    def reparameterize(self, mu, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) * temperature
        return mu + eps * std

    def decode(self, z, joint_type_id, kl, skips=None):
        cond = self.get_cond(joint_type_id, kl)
        h = torch.cat([z, cond], dim=1)
        h = self.fc_dec(h)
        h = h.view(-1, self.base_ch * 8, 8, 8)
        h = self.dec_bottleneck(h)
        h = self.attn(h)

        if skips is None:
            b = h.size(0)
            device = h.device
            s1 = torch.zeros(b, self.base_ch, 64, 64, device=device)
            s2 = torch.zeros(b, self.base_ch * 2, 32, 32, device=device)
            s3 = torch.zeros(b, self.base_ch * 4, 16, 16, device=device)
        else:
            s1, s2, s3 = skips

        h = self.up1(h, s3)
        h = self.up2(h, s2)
        h = self.up3(h, s1)

        h = self.up4(h)
        h = self.final_refine(h)
        x_recon = self.final_out(h)
        return x_recon

    def forward(self, x, joint_type_id, kl):
        mu, logvar, skips = self.encode(x, joint_type_id, kl)
        z = self.reparameterize(mu, logvar, temperature=1.0)
        recon = self.decode(z, joint_type_id, kl, skips=skips)
        return recon, mu, logvar


# ============================================================
# 12. Loss
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
# 13. Build augmented training items
# ============================================================
def build_augmented_train_items(train_items, max_aug_ratio=0.5, aug_types=None):
    if aug_types is None:
        aug_types = ["flip", "rot_left", "rot_right"]

    original_items = []
    for item in train_items:
        x = dict(item)
        x["aug_type"] = "none"
        original_items.append(x)

    n_original = len(original_items)
    max_aug = int(n_original * max_aug_ratio)

    if max_aug <= 0:
        print(f"[Augment] original train images: {n_original}")
        print(f"[Augment] added augmented images: 0")
        print(f"[Augment] final train size: {n_original}")
        return original_items

    aug_candidates = []
    for item in train_items:
        for aug_type in aug_types:
            x = dict(item)
            x["aug_type"] = aug_type
            aug_candidates.append(x)

    random.shuffle(aug_candidates)
    aug_items = aug_candidates[:max_aug]

    print(f"[Augment] original train images: {n_original}")
    print(f"[Augment] added augmented images: {len(aug_items)}")
    print(f"[Augment] final train size: {n_original + len(aug_items)}")

    return original_items + aug_items


# ============================================================
# 14. Build weighted sampler
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
# 15. One epoch runner
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
        imgs = imgs.to(DEVICE, non_blocking=True)
        joint_type_ids = joint_type_ids.to(DEVICE, non_blocking=True)
        kls = kls.to(DEVICE, non_blocking=True)

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
# 16. Save recon preview
# ============================================================
@torch.no_grad()
def save_recon_examples(model, loader, out_path, max_samples=8):
    model.eval()

    imgs_all = []
    recon_all = []

    for imgs, joint_type_ids, kls, _ in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        joint_type_ids = joint_type_ids.to(DEVICE, non_blocking=True)
        kls = kls.to(DEVICE, non_blocking=True)

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
# 17. Build data
# ============================================================
metadata = load_metadata()
all_items, skipped = build_items(metadata)

if len(all_items) == 0:
    raise RuntimeError("No usable items found. Please check filenames and hand.xlsx.")

all_indices = list(range(len(all_items)))
val_size = int(len(all_indices) * cfg.VAL_RATIO)
train_size = len(all_indices) - val_size

train_idx_subset, val_idx_subset = random_split(
    all_indices,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_indices = list(train_idx_subset)
val_indices = list(val_idx_subset)

train_items = [all_items[i] for i in train_indices]
val_items = [all_items[i] for i in val_indices]

train_items_augmented = build_augmented_train_items(
    train_items=train_items,
    max_aug_ratio=cfg.AUG_MAX_RATIO,
    aug_types=cfg.AUG_TYPES
)

val_items_clean = []
for item in val_items:
    x = dict(item)
    x["aug_type"] = "none"
    val_items_clean.append(x)

PIN_MEMORY = torch.cuda.is_available()

train_dataset = JointKLDataset(train_items_augmented, img_size=cfg.IMG_SIZE)
val_dataset = JointKLDataset(val_items_clean, img_size=cfg.IMG_SIZE)

sampler, sampling_weights = build_weighted_sampler(train_items_augmented)

train_loader = DataLoader(
    train_dataset,
    batch_size=cfg.BATCH_SIZE,
    sampler=sampler,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(cfg.NUM_WORKERS > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=cfg.BATCH_SIZE,
    shuffle=False,
    num_workers=cfg.NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=(cfg.NUM_WORKERS > 0),
)


# ============================================================
# 18. Model / optimizer
# ============================================================
model = DualConditionCVAE(
    img_size=cfg.IMG_SIZE,
    latent_dim=cfg.LATENT_DIM,
    num_joint_types=NUM_JOINT_TYPES,
    num_kl_classes=NUM_KL_CLASSES,
    joint_emb_dim=cfg.JOINT_EMB_DIM,
    kl_emb_dim=cfg.KL_EMB_DIM,
    base_ch=cfg.BASE_CH
)

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    print(f"[Using DataParallel on {torch.cuda.device_count()} GPUs]")
    model = nn.DataParallel(model)

model = model.to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg.LR,
    weight_decay=cfg.WEIGHT_DECAY
)


# ============================================================
# 19. Training
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
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state_dict, best_model_path)
        print(f"  -> saved best model to {best_model_path}")

history_path = os.path.join(cfg.OUTPUT_DIR, "training_history.json")
save_json(history, history_path)

summary = {
    "num_total": len(all_items),
    "num_train_original": len(train_items),
    "num_train_after_aug": len(train_items_augmented),
    "num_val": len(val_items),
    "best_val_loss": best_val_loss,
    "best_model_path": best_model_path,
    "sampling_weights_preview": sampling_weights[:20],
}
save_json(summary, os.path.join(cfg.OUTPUT_DIR, "summary.json"))


# ============================================================
# 20. Reload best
# ============================================================
state = torch.load(best_model_path, map_location=DEVICE)
if isinstance(model, nn.DataParallel):
    model.module.load_state_dict(state)
else:
    model.load_state_dict(state)
model.eval()

core_model = model.module if isinstance(model, nn.DataParallel) else model
print("[Model wrapper]", "DataParallel" if isinstance(model, nn.DataParallel) else "Single GPU / CPU")


# ============================================================
# 21. Collect latent bank
# ============================================================
@torch.no_grad()
def collect_latent_bank(core_model, items):
    bank = defaultdict(list)
    meta_bank = defaultdict(list)

    infer_items = []
    for item in items:
        x = dict(item)
        x["aug_type"] = "none"
        infer_items.append(x)

    infer_ds = JointKLDataset(infer_items, img_size=cfg.IMG_SIZE)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    for imgs, joint_type_ids, kls, metas in infer_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        joint_type_ids = joint_type_ids.to(DEVICE, non_blocking=True)
        kls = kls.to(DEVICE, non_blocking=True)

        mu, logvar, _ = core_model.encode(imgs, joint_type_ids, kls)

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


# ============================================================
# 22. Collect prototype skip bank
# ============================================================
@torch.no_grad()
def collect_prototype_skip_bank(core_model, items, max_per_key=64):
    skip_bank = defaultdict(list)

    infer_items = []
    for item in items:
        x = dict(item)
        x["aug_type"] = "none"
        infer_items.append(x)

    infer_ds = JointKLDataset(infer_items, img_size=cfg.IMG_SIZE)
    infer_loader = DataLoader(
        infer_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(cfg.NUM_WORKERS > 0),
    )

    for imgs, joint_type_ids, kls, _ in infer_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        joint_type_ids = joint_type_ids.to(DEVICE, non_blocking=True)
        kls = kls.to(DEVICE, non_blocking=True)

        s1, s2, s3, _ = core_model.encode_features(imgs)

        for i in range(imgs.size(0)):
            key = (int(joint_type_ids[i].item()), int(kls[i].item()))
            if len(skip_bank[key]) < max_per_key:
                skip_bank[key].append((
                    s1[i].detach().cpu(),
                    s2[i].detach().cpu(),
                    s3[i].detach().cpu()
                ))

    return skip_bank


latent_bank, latent_meta_bank = collect_latent_bank(core_model, all_items)
prototype_skip_bank = collect_prototype_skip_bank(core_model, all_items, max_per_key=64)


# ============================================================
# 23. Generation
# ============================================================
@torch.no_grad()
def generate_from_prior(core_model, joint_type_name, kl, n=16, temperature=1.0):
    jt_id = JOINT_TYPE_TO_ID[joint_type_name]
    z = torch.randn(n, cfg.LATENT_DIM, device=DEVICE) * temperature
    jt_tensor = torch.full((n,), jt_id, dtype=torch.long, device=DEVICE)
    kl_tensor = torch.full((n,), kl, dtype=torch.long, device=DEVICE)
    fake = core_model.decode(z, jt_tensor, kl_tensor, skips=None)
    return fake


@torch.no_grad()
def generate_real_latent_guided(core_model, joint_type_name, kl, n=16, noise_std=0.08):
    jt_id = JOINT_TYPE_TO_ID[joint_type_name]
    key = (jt_id, kl)

    if key not in latent_bank or latent_bank[key] is None or len(latent_bank[key]) == 0:
        z = torch.randn(n, cfg.LATENT_DIM, device=DEVICE)
    else:
        bank = latent_bank[key].to(DEVICE)
        idx = torch.randint(0, bank.size(0), (n,), device=DEVICE)
        z_real = bank[idx]
        noise = torch.randn_like(z_real) * noise_std
        z = z_real + noise

    jt_tensor = torch.full((n,), jt_id, dtype=torch.long, device=DEVICE)
    kl_tensor = torch.full((n,), kl, dtype=torch.long, device=DEVICE)

    skips = None
    if key in prototype_skip_bank and len(prototype_skip_bank[key]) > 0:
        selected = [
            prototype_skip_bank[key][random.randint(0, len(prototype_skip_bank[key]) - 1)]
            for _ in range(n)
        ]
        s1 = torch.stack([x[0] for x in selected], dim=0).to(DEVICE)
        s2 = torch.stack([x[1] for x in selected], dim=0).to(DEVICE)
        s3 = torch.stack([x[2] for x in selected], dim=0).to(DEVICE)
        skips = (s1, s2, s3)

    fake = core_model.decode(z, jt_tensor, kl_tensor, skips=skips)
    return fake


@torch.no_grad()
def save_generated_samples(core_model, targets, n_per_target=50):
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
                    core_model=core_model,
                    joint_type_name=joint_type_name,
                    kl=kl,
                    n=cur_n,
                    noise_std=cfg.GUIDED_NOISE_STD
                )
            else:
                fake = generate_from_prior(
                    core_model=core_model,
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
    core_model=core_model,
    targets=cfg.GENERATE_TARGETS,
    n_per_target=cfg.N_SAMPLES_PER_TARGET
)


# ============================================================
# 24. Latent interpolation
# ============================================================
@torch.no_grad()
def latent_interpolation(core_model, joint_type_name="DIP", kl=4, steps=8, out_path="interp.png"):
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

    skips = None
    if key in prototype_skip_bank and len(prototype_skip_bank[key]) > 0:
        proto = prototype_skip_bank[key][0]
        s1 = proto[0].unsqueeze(0).repeat(steps, 1, 1, 1).to(DEVICE)
        s2 = proto[1].unsqueeze(0).repeat(steps, 1, 1, 1).to(DEVICE)
        s3 = proto[2].unsqueeze(0).repeat(steps, 1, 1, 1).to(DEVICE)
        skips = (s1, s2, s3)

    imgs = core_model.decode(z_all, jt_tensor, kl_tensor, skips=skips).cpu()
    save_image_grid(imgs, out_path, nrow=steps)
    print(f"[Saved interpolation] {out_path}")


latent_interpolation(
    core_model=core_model,
    joint_type_name="DIP",
    kl=4,
    steps=8,
    out_path=os.path.join(cfg.OUTPUT_DIR, "latent_interp_DIP_KL4.png")
)

print("\n[All Done] Sharp U-Net dual-condition CVAE training and generation completed.")