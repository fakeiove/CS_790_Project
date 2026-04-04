# ============================================================
# PIP2-5 KL Classification + Joint Difference Validation
# ============================================================
# Functions included:
# 1) patient-level split
# 2) pooled KL classifier
# 3) pooled KL classifier + joint index
# 4) joint classifier (predict pip2/pip3/pip4/pip5)
# 5) feature extraction
# 6) PCA / UMAP visualization
# 7) ANOVA / Kruskal-Wallis statistical tests
#
# Author: ChatGPT
# ============================================================

import os
import re
import random
import warnings
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from scipy.stats import f_oneway, kruskal

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Optional UMAP
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision import models

# -----------------------------
# 0. Global Config
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("[Device]", DEVICE)

# -----------------------------
# 1. User Config
# -----------------------------
CONFIG = {
    "image_dir": "Finger Joints",
    "excel_path": "hand.xlsx",
    "id_col": "id",  # <-- change if needed

    "joint_to_kl_col": {
        "pip2": "v00PIP2_KL",
        "pip3": "v00PIP3_KL",
        "pip4": "v00PIP4_KL",
        "pip5": "v00PIP5_KL",
    },

    "valid_joints": ["pip2", "pip3", "pip4", "pip5",
                     "dip2", "dip3", "dip4", "dip5",
                     "mcp2", "mcp3", "mcp4", "mcp5"],
    "valid_kl": [0, 1, 2, 3, 4],

    "img_size": 128,    # Match our VAE/diffusion pipeline
    "batch_size": 32,
    "num_workers": 0,
    "epochs": 12,
    "lr": 1e-4,
    "weight_decay": 1e-4,

    "test_size": 0.15,
    "val_size": 0.15,
    "use_weighted_sampler": True,

    "feature_batch_size": 64,
    "num_joint_classes": 4,
    "num_kl_classes": 5,

    "save_dir": "exp_outputs"
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

# -----------------------------
# 2. Utility Functions
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def parse_filename(filename):
    """
    Expected pattern like:
    9000099_pip2.png
    returns (patient_id, joint)
    """
    name = os.path.basename(filename).lower()
    m = re.match(r"^(\d+)_([a-z0-9]+)\.(png|jpg|jpeg|bmp|tif|tiff)$", name)
    if not m:
        return None, None
    patient_id = m.group(1)
    joint = m.group(2)
    return patient_id, joint


def safe_int(x):
    try:
        if pd.isna(x):
            return None
        return int(x)
    except:
        return None


# -----------------------------
# 3. Build Metadata
# -----------------------------
def build_metadata(config):
    image_dir = config["image_dir"]
    excel_path = config["excel_path"]
    id_col = config["id_col"]
    joint_to_kl_col = config["joint_to_kl_col"]
    valid_joints = set(config["valid_joints"])
    valid_kl = set(config["valid_kl"])

    df = pd.read_excel(excel_path)

    # normalize Excel ID to string
    df[id_col] = df[id_col].astype(str).str.extract(r"(\d+)")[0]

    # build quick lookup: patient_id -> row
    df_map = {}
    for _, row in df.iterrows():
        pid = str(row[id_col])
        df_map[pid] = row

    records = []
    skipped = []

    for fname in os.listdir(image_dir):
        fpath = os.path.join(image_dir, fname)
        if not os.path.isfile(fpath):
            continue

        patient_id, joint = parse_filename(fname)
        if patient_id is None:
            skipped.append((fname, "filename_parse_failed"))
            continue

        if joint not in valid_joints:
            skipped.append((fname, "joint_not_in_target_pip2_5"))
            continue

        if patient_id not in df_map:
            skipped.append((fname, "id_not_found_in_excel"))
            continue

        row = df_map[patient_id]

        if joint not in joint_to_kl_col:
            skipped.append((fname, "joint_missing_mapping"))
            continue

        kl_col = joint_to_kl_col[joint]
        if kl_col not in df.columns:
            skipped.append((fname, f"excel_missing_column_{kl_col}"))
            continue

        kl = safe_int(row[kl_col])
        if kl is None:
            skipped.append((fname, "kl_missing"))
            continue

        if kl not in valid_kl:
            skipped.append((fname, f"kl_not_in_{sorted(valid_kl)}"))
            continue

        joint_idx = config["valid_joints"].index(joint)

        records.append({
            "patient_id": patient_id,
            "image_path": fpath,
            "filename": fname,
            "joint": joint,
            "joint_idx": joint_idx,
            "kl": kl
        })

    meta = pd.DataFrame(records)

    print("[Metadata] usable images:", len(meta))
    print("[Metadata] skipped images:", len(skipped))
    if len(skipped) > 0:
        print("[Metadata] first few skipped files:", skipped[:10])

    if len(meta) > 0:
        print("\n[Metadata] KL distribution:")
        print(meta["kl"].value_counts().sort_index())

        print("\n[Metadata] Joint distribution:")
        print(meta["joint"].value_counts())

        print("\n[Metadata] Joint x KL table:")
        print(pd.crosstab(meta["joint"], meta["kl"]))

    return meta, skipped


# -----------------------------
# 4. Patient-level Split
# -----------------------------
def patient_level_split(meta, test_size=0.15, val_size=0.15, seed=42):
    """
    Split by patient_id to avoid leakage.
    val_size is fraction of total data.
    """
    patients = meta["patient_id"].unique().tolist()

    train_patients, test_patients = train_test_split(
        patients, test_size=test_size, random_state=seed
    )

    train_val_fraction = 1.0 - test_size
    val_relative = val_size / train_val_fraction

    train_patients, val_patients = train_test_split(
        train_patients, test_size=val_relative, random_state=seed
    )

    train_df = meta[meta["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = meta[meta["patient_id"].isin(val_patients)].reset_index(drop=True)
    test_df = meta[meta["patient_id"].isin(test_patients)].reset_index(drop=True)

    print("\n[Split Summary]")
    print("Train patients:", len(train_patients), "images:", len(train_df))
    print("Val patients  :", len(val_patients), "images:", len(val_df))
    print("Test patients :", len(test_patients), "images:", len(test_df))

    overlap = (
        set(train_df["patient_id"]) & set(val_df["patient_id"]) |
        set(train_df["patient_id"]) & set(test_df["patient_id"]) |
        set(val_df["patient_id"]) & set(test_df["patient_id"])
    )
    print("Patient overlap:", len(overlap))

    return train_df, val_df, test_df


# -----------------------------
# 5. Dataset
# -----------------------------
class HandJointDataset(Dataset):
    def __init__(self, df, transform=None, target_type="kl", use_joint_input=False):
        """
        target_type:
            - "kl"
            - "joint"
        use_joint_input:
            - False: return (image, target)
            - True : return (image, joint_idx, target)
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.target_type = target_type
        self.use_joint_input = use_joint_input

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["image_path"]).convert("L")

        if self.transform:
            img = self.transform(img)

        if self.target_type == "kl":
            target = int(row["kl"])
        elif self.target_type == "joint":
            target = int(row["joint_idx"])
        else:
            raise ValueError("target_type must be 'kl' or 'joint'")

        joint_idx = int(row["joint_idx"])

        if self.use_joint_input:
            return img, torch.tensor(joint_idx, dtype=torch.long), torch.tensor(target, dtype=torch.long)
        else:
            return img, torch.tensor(target, dtype=torch.long)


def get_transforms(img_size=224):
    train_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=8),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])

    eval_tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5]),
    ])
    return train_tf, eval_tf


def make_weighted_sampler(df, label_col):
    counts = df[label_col].value_counts().to_dict()
    sample_weights = [1.0 / counts[label] for label in df[label_col]]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler


def build_dataloaders(train_df, val_df, test_df, config, target_type="kl", use_joint_input=False):
    train_tf, eval_tf = get_transforms(config["img_size"])

    train_ds = HandJointDataset(train_df, transform=train_tf, target_type=target_type, use_joint_input=use_joint_input)
    val_ds = HandJointDataset(val_df, transform=eval_tf, target_type=target_type, use_joint_input=use_joint_input)
    test_ds = HandJointDataset(test_df, transform=eval_tf, target_type=target_type, use_joint_input=use_joint_input)

    sampler = None
    if config["use_weighted_sampler"] and target_type == "kl":
        sampler = make_weighted_sampler(train_df, "kl")

    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=config["num_workers"]
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    return train_loader, val_loader, test_loader


# -----------------------------
# 6. Models
# -----------------------------
def build_backbone(num_channels=1, pretrained=True):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)

    # adapt first conv to grayscale
    old_conv = model.conv1
    model.conv1 = nn.Conv2d(
        in_channels=num_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False
    )

    with torch.no_grad():
        if pretrained:
            model.conv1.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

    return model


class PooledKLClassifier(nn.Module):
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = build_backbone(num_channels=1, pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


class PooledKLWithJointClassifier(nn.Module):
    def __init__(self, num_classes=5, num_joints=4, joint_emb_dim=8, pretrained=True):
        super().__init__()
        self.backbone = build_backbone(num_channels=1, pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.joint_emb = nn.Embedding(num_joints, joint_emb_dim)

        self.classifier = nn.Sequential(
            nn.Linear(in_features + joint_emb_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, joint_idx):
        feat = self.backbone(x)
        jemb = self.joint_emb(joint_idx)
        out = torch.cat([feat, jemb], dim=1)
        return self.classifier(out)


class JointClassifier(nn.Module):
    def __init__(self, num_joint_classes=4, pretrained=True):
        super().__init__()
        self.backbone = build_backbone(num_channels=1, pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_joint_classes)

    def forward(self, x):
        return self.backbone(x)


# -----------------------------
# 7. Train / Eval
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion, device, use_joint_input=False):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        optimizer.zero_grad()

        if use_joint_input:
            images, joint_idx, targets = batch
            images = images.to(device)
            joint_idx = joint_idx.to(device)
            targets = targets.to(device)
            logits = model(images, joint_idx)
        else:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)

        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * targets.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_joint_input=False):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in loader:
        if use_joint_input:
            images, joint_idx, targets = batch
            images = images.to(device)
            joint_idx = joint_idx.to(device)
            targets = targets.to(device)
            logits = model(images, joint_idx)
        else:
            images, targets = batch
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)

        loss = criterion(logits, targets)

        total_loss += loss.item() * targets.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1, np.array(all_targets), np.array(all_preds)


def run_training(
    model,
    train_loader,
    val_loader,
    test_loader,
    config,
    save_name="model.pt",
    use_joint_input=False
):
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_val_f1 = -1
    best_path = os.path.join(config["save_dir"], save_name)

    history = []

    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE, use_joint_input=use_joint_input
        )
        va_loss, va_acc, va_f1, _, _ = evaluate(
            model, val_loader, criterion, DEVICE, use_joint_input=use_joint_input
        )

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "train_f1": tr_f1,
            "val_loss": va_loss,
            "val_acc": va_acc,
            "val_f1": va_f1
        })

        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} train_f1={tr_f1:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f} val_f1={va_f1:.4f}"
        )

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            torch.save(model.state_dict(), best_path)

    print(f"[Best model saved] {best_path} | best_val_f1={best_val_f1:.4f}")

    # load best
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    te_loss, te_acc, te_f1, y_true, y_pred = evaluate(
        model, test_loader, criterion, DEVICE, use_joint_input=use_joint_input
    )

    print("\n[Test Result]")
    print(f"test_loss={te_loss:.4f} test_acc={te_acc:.4f} test_f1={te_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config["save_dir"], save_name.replace(".pt", "_history.csv")), index=False)

    return model, history_df, {
        "test_loss": te_loss,
        "test_acc": te_acc,
        "test_f1_macro": te_f1,
        "y_true": y_true,
        "y_pred": y_pred
    }


# -----------------------------
# 8. Feature Extraction
# -----------------------------
class FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = build_backbone(num_channels=1, pretrained=pretrained)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.out_dim = in_features

    def forward(self, x):
        return self.backbone(x)


@torch.no_grad()
def extract_features(df, config):
    _, eval_tf = get_transforms(config["img_size"])
    ds = HandJointDataset(df, transform=eval_tf, target_type="kl", use_joint_input=False)
    loader = DataLoader(
        ds,
        batch_size=config["feature_batch_size"],
        shuffle=False,
        num_workers=config["num_workers"]
    )

    extractor = FeatureExtractor(pretrained=True).to(DEVICE)
    extractor.eval()

    feats = []
    rows = []

    for batch_idx, batch in enumerate(loader):
        images, targets = batch
        images = images.to(DEVICE)
        emb = extractor(images).cpu().numpy()
        feats.append(emb)

    feats = np.concatenate(feats, axis=0)
    out_df = df.copy().reset_index(drop=True)
    return feats, out_df


# -----------------------------
# 9. Visualization
# -----------------------------
def plot_pca(features, meta_df, save_path):
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pca = PCA(n_components=2, random_state=SEED)
    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    joints = meta_df["joint"].unique().tolist()
    for joint in joints:
        mask = (meta_df["joint"] == joint).values
        plt.scatter(Z[mask, 0], Z[mask, 1], alpha=0.6, label=joint)

    plt.title("PCA of deep features (colored by joint)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("[Saved]", save_path)


def plot_umap(features, meta_df, save_path):
    if not HAS_UMAP:
        print("UMAP not installed. Skip UMAP plot.")
        return

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    reducer = umap.UMAP(
        n_components=2,
        random_state=SEED,
        n_neighbors=15,
        min_dist=0.1
    )
    Z = reducer.fit_transform(X)

    plt.figure(figsize=(8, 6))
    joints = meta_df["joint"].unique().tolist()
    for joint in joints:
        mask = (meta_df["joint"] == joint).values
        plt.scatter(Z[mask, 0], Z[mask, 1], alpha=0.6, label=joint)

    plt.title("UMAP of deep features (colored by joint)")
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()

    print("[Saved]", save_path)


# -----------------------------
# 10. Statistical Tests
# -----------------------------
def run_joint_difference_tests(features, meta_df):
    """
    Use top PCA components as scalar summaries, then compare across joints.
    Joint groups: pip2/pip3/pip4/pip5
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    pca = PCA(n_components=min(10, X.shape[1]), random_state=SEED)
    Z = pca.fit_transform(X)

    result_rows = []
    joints = sorted(meta_df["joint"].unique().tolist())

    print("\n[Statistical Test: joint difference on PCA components]")
    for comp_idx in range(min(5, Z.shape[1])):
        values_by_group = []
        for joint in joints:
            vals = Z[meta_df["joint"] == joint, comp_idx]
            values_by_group.append(vals)

        # ANOVA
        try:
            anova_stat, anova_p = f_oneway(*values_by_group)
        except:
            anova_stat, anova_p = np.nan, np.nan

        # Kruskal-Wallis
        try:
            kw_stat, kw_p = kruskal(*values_by_group)
        except:
            kw_stat, kw_p = np.nan, np.nan

        result_rows.append({
            "component": f"PC{comp_idx + 1}",
            "anova_stat": anova_stat,
            "anova_p": anova_p,
            "kruskal_stat": kw_stat,
            "kruskal_p": kw_p
        })

    result_df = pd.DataFrame(result_rows)
    print(result_df)
    result_df.to_csv(os.path.join(CONFIG["save_dir"], "joint_difference_stats.csv"), index=False)

    return result_df


# -----------------------------
# 11. Experiment Runners
# -----------------------------
def run_pooled_kl_experiment(train_df, val_df, test_df, config):
    print("\n" + "=" * 70)
    print("Experiment 1: Pooled KL Classifier")
    print("=" * 70)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config, target_type="kl", use_joint_input=False
    )

    model = PooledKLClassifier(num_classes=config["num_kl_classes"], pretrained=True)

    return run_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        save_name="pooled_kl_classifier.pt",
        use_joint_input=False
    )


def run_pooled_plus_joint_experiment(train_df, val_df, test_df, config):
    print("\n" + "=" * 70)
    print("Experiment 2: Pooled KL Classifier + Joint Index")
    print("=" * 70)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config, target_type="kl", use_joint_input=True
    )

    model = PooledKLWithJointClassifier(
        num_classes=config["num_kl_classes"],
        num_joints=len(config["valid_joints"]),
        joint_emb_dim=8,
        pretrained=True
    )

    return run_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        save_name="pooled_plus_joint_kl_classifier.pt",
        use_joint_input=True
    )


def run_joint_classifier_experiment(train_df, val_df, test_df, config):
    print("\n" + "=" * 70)
    print("Experiment 3: Joint Classifier (predict pip2/pip3/pip4/pip5)")
    print("=" * 70)

    train_loader, val_loader, test_loader = build_dataloaders(
        train_df, val_df, test_df, config, target_type="joint", use_joint_input=False
    )

    model = JointClassifier(num_joint_classes=config["num_joint_classes"], pretrained=True)

    return run_training(
        model,
        train_loader,
        val_loader,
        test_loader,
        config,
        save_name="joint_classifier.pt",
        use_joint_input=False
    )


def run_per_joint_kl_experiments(train_df, val_df, test_df, config):
    print("\n" + "=" * 70)
    print("Experiment 4: Per-joint KL Classifiers")
    print("=" * 70)

    summary = []

    for joint in config["valid_joints"]:
        tr = train_df[train_df["joint"] == joint].reset_index(drop=True)
        va = val_df[val_df["joint"] == joint].reset_index(drop=True)
        te = test_df[test_df["joint"] == joint].reset_index(drop=True)

        if len(tr) < 10 or len(va) < 5 or len(te) < 5:
            print(f"[Skip {joint}] too few samples")
            continue

        print(f"\n--- Per-joint KL experiment: {joint} ---")
        train_loader, val_loader, test_loader = build_dataloaders(
            tr, va, te, config, target_type="kl", use_joint_input=False
        )

        model = PooledKLClassifier(num_classes=config["num_kl_classes"], pretrained=True)

        _, _, result = run_training(
            model,
            train_loader,
            val_loader,
            test_loader,
            config,
            save_name=f"{joint}_kl_classifier.pt",
            use_joint_input=False
        )

        summary.append({
            "joint": joint,
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
            "test_acc": result["test_acc"],
            "test_f1_macro": result["test_f1_macro"]
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(config["save_dir"], "per_joint_kl_summary.csv"), index=False)
    print("\n[Per-joint KL Summary]")
    print(summary_df)
    return summary_df


# -----------------------------
# 12. Main
# -----------------------------
def main():
    meta, skipped = build_metadata(CONFIG)
    if len(meta) == 0:
        print("No usable images found. Please check filenames / excel column names.")
        return

    train_df, val_df, test_df = patient_level_split(
        meta,
        test_size=CONFIG["test_size"],
        val_size=CONFIG["val_size"],
        seed=SEED
    )

    # 1) pooled KL classifier
    pooled_model, pooled_hist, pooled_result = run_pooled_kl_experiment(train_df, val_df, test_df, CONFIG)

    # 2) pooled KL + joint info
    pooled_joint_model, pooled_joint_hist, pooled_joint_result = run_pooled_plus_joint_experiment(train_df, val_df, test_df, CONFIG)

    # 3) joint classifier
    joint_model, joint_hist, joint_result = run_joint_classifier_experiment(train_df, val_df, test_df, CONFIG)

    # 4) per-joint KL experiments
    per_joint_summary = run_per_joint_kl_experiments(train_df, val_df, test_df, CONFIG)

    # 5) feature extraction on all data
    print("\n" + "=" * 70)
    print("Feature extraction + PCA/UMAP + statistical tests")
    print("=" * 70)

    features, feature_meta = extract_features(meta, CONFIG)
    np.save(os.path.join(CONFIG["save_dir"], "deep_features.npy"), features)
    feature_meta.to_csv(os.path.join(CONFIG["save_dir"], "feature_meta.csv"), index=False)

    # 6) PCA / UMAP
    plot_pca(features, feature_meta, os.path.join(CONFIG["save_dir"], "pca_joint_plot.png"))
    plot_umap(features, feature_meta, os.path.join(CONFIG["save_dir"], "umap_joint_plot.png"))

    # 7) statistical tests
    stats_df = run_joint_difference_tests(features, feature_meta)

    # 8) overall summary
    summary_rows = [
        {
            "experiment": "pooled_kl_classifier",
            "test_acc": pooled_result["test_acc"],
            "test_f1_macro": pooled_result["test_f1_macro"]
        },
        {
            "experiment": "pooled_kl_plus_joint_classifier",
            "test_acc": pooled_joint_result["test_acc"],
            "test_f1_macro": pooled_joint_result["test_f1_macro"]
        },
        {
            "experiment": "joint_classifier",
            "test_acc": joint_result["test_acc"],
            "test_f1_macro": joint_result["test_f1_macro"]
        }
    ]

    overall_summary = pd.DataFrame(summary_rows)
    overall_summary.to_csv(os.path.join(CONFIG["save_dir"], "overall_summary.csv"), index=False)

    print("\n" + "=" * 70)
    print("Overall Summary")
    print("=" * 70)
    print(overall_summary)

    print("\nInterpretation guide:")
    print("1. If joint_classifier test_acc is near chance (~0.25), PIP2-5 are hard to distinguish.")
    print("2. If pooled_kl_plus_joint only slightly improves over pooled_kl, joint info matters little.")
    print("3. If pooled_kl performs as well as or better than per-joint KL models, pooling is justified.")
    print("4. If PCA/UMAP shows heavy overlap and stats are weak/non-significant, joint difference is limited.")


if __name__ == "__main__":
    main()