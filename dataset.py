"""
Dataset classes for loading hand joint X-ray ROI images with KL grades.

Key changes from original:
  1. 70/15/15 train/val/test split (was 90/10 with no test set)
  2. Stratified by KL grade AND split by patient_id (no data leakage)
  3. Split indices saved to JSON for reproducibility
  4. All downstream code (VAE, Diffusion, Generate) uses ONLY train split
  5. Val split for hyperparameter tuning during training
  6. Test split reserved for FINAL evaluation only

Image naming convention: {duryeaid}_{joint_lower}.png
Example: 9004905_pip2.png, 9005321_dip3.png
"""

import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# Data Splitting
# ============================================================

def create_patient_split(csv_path, joint_types=['DIP'], visit='v00',
                         train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                         seed=42, save_path=None):
    """
    Split patients into train/val/test sets.

    Strategy:
      1. Group all joints by patient_id
      2. Assign each patient a "primary KL" = max KL grade across their joints
         (ensures patients with rare KL3/4 are properly stratified)
      3. Within each primary KL stratum, randomly assign patients to train/val/test
      4. Save the split to JSON for reproducibility

    This guarantees:
      - No patient appears in more than one split (no data leakage)
      - Each split has proportional representation of all KL grades
      - The split is deterministic given the same seed
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    df = pd.read_csv(csv_path)
    df = df[df['joint_type'].isin(joint_types)].copy()

    kl_col = f'{visit}_KL'

    # Group by patient, get their "primary KL" = max KL across all their joints
    patient_max_kl = df.groupby('patient_id')[kl_col].max().reset_index()
    patient_max_kl.columns = ['patient_id', 'primary_kl']

    rng = np.random.RandomState(seed)

    train_patients = []
    val_patients = []
    test_patients = []

    for kl_grade in sorted(patient_max_kl['primary_kl'].unique()):
        patients_in_stratum = patient_max_kl[
            patient_max_kl['primary_kl'] == kl_grade
        ]['patient_id'].values.copy()

        rng.shuffle(patients_in_stratum)

        n = len(patients_in_stratum)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_patients.extend(patients_in_stratum[:n_train].tolist())
        val_patients.extend(patients_in_stratum[n_train:n_train + n_val].tolist())
        test_patients.extend(patients_in_stratum[n_train + n_val:].tolist())

    split = {
        'train': sorted(train_patients),
        'val': sorted(val_patients),
        'test': sorted(test_patients),
        'config': {
            'joint_types': joint_types,
            'visit': visit,
            'train_ratio': train_ratio,
            'val_ratio': val_ratio,
            'test_ratio': test_ratio,
            'seed': seed,
            'num_train_patients': len(train_patients),
            'num_val_patients': len(val_patients),
            'num_test_patients': len(test_patients),
        }
    }

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(split, f, indent=2)
        print(f"Split saved to {save_path}")

    _print_split_summary(df, split, kl_col)

    return split


def load_patient_split(split_path):
    """Load a previously saved patient split from JSON."""
    with open(split_path, 'r') as f:
        split = json.load(f)
    print(f"Loaded split from {split_path}")
    print(f"  Train: {len(split['train'])} patients | "
          f"Val: {len(split['val'])} patients | "
          f"Test: {len(split['test'])} patients")
    return split


def _print_split_summary(df, split, kl_col):
    """Print KL grade distribution for each split."""
    print(f"\n{'='*60}")
    print(f"Data Split Summary (70 / 15 / 15)")
    print(f"{'='*60}")

    for split_name in ['train', 'val', 'test']:
        patient_ids = split[split_name]
        subset = df[df['patient_id'].isin(patient_ids)]
        total = len(subset)

        print(f"\n  {split_name.upper()} — {len(patient_ids)} patients, {total} joints:")
        if total > 0:
            for kl in sorted(subset[kl_col].unique()):
                count = (subset[kl_col] == kl).sum()
                print(f"    KL{int(kl)}: {count:>6d} ({count/total*100:>5.1f}%)")

    print(f"\n{'='*60}\n")


# [FIXED] Added missing get_kl_distribution function
def get_kl_distribution(csv_path, joint_types, visit):
    """Print KL grade distribution for a given visit."""
    df = pd.read_csv(csv_path)
    df = df[df['joint_type'].isin(joint_types)].copy()
    kl_col = f'{visit}_KL'

    print(f"\nKL Distribution for {visit} ({', '.join(joint_types)}):")
    total = len(df)
    for kl in sorted(df[kl_col].dropna().unique()):
        count = (df[kl_col] == kl).sum()
        print(f"  KL{int(kl)}: {count:>6d} ({count/total*100:>5.1f}%)")
    print(f"  Total: {total}")


# ============================================================
# Dataset Classes
# ============================================================

class HandJointDataset(Dataset):
    """
    Dataset for hand joint ROI images with KL grade labels.

    Args:
        csv_path: Path to hand_long_clean2.csv
        image_dir: Directory containing ROI PNG images
        patient_ids: List of patient IDs to include (from split).
                     If None, uses ALL patients (not recommended).
        joint_types: List of joint types to include
        visit: 'v00' or 'v06'
        img_size: Target image size
        kl_filter: Optional list of KL grades to include
        augment: Whether to apply data augmentation
    """
    def __init__(self, csv_path, image_dir, patient_ids=None,
                 joint_types=['DIP'], visit='v00', img_size=128,
                 kl_filter=None, augment=False):
        self.image_dir = image_dir
        self.img_size = img_size
        self.visit = visit

        df = pd.read_csv(csv_path)
        df = df[df['joint_type'].isin(joint_types)].copy()

        if patient_ids is not None:
            df = df[df['patient_id'].isin(patient_ids)]

        df['filename'] = df['patient_id'].astype(str) + '_' + df['joint'].str.lower() + '.png'
        df['kl_grade'] = df[f'{visit}_KL']

        if kl_filter is not None:
            df = df[df['kl_grade'].isin(kl_filter)]

        df['filepath'] = df['filename'].apply(lambda f: os.path.join(image_dir, f))
        df = df[df['filepath'].apply(os.path.exists)].reset_index(drop=True)

        self.data = df[['filepath', 'kl_grade', 'filename', 'patient_id',
                        'duryeaid', 'joint', 'joint_type']].copy()

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(10),
                transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row['filepath']).convert('L')
        img = self.transform(img)
        kl_grade = int(row['kl_grade'])
        return {
            'image': img,
            'kl_grade': kl_grade,
            'filename': row['filename']
        }


class PairedProgressionDataset(Dataset):
    """
    Dataset for paired v00->v06 progression images.
    Only includes pairs where KL grade increased (real disease progression).
    Used for fine-tuning the diffusion model.
    """
    def __init__(self, csv_path, image_dir_v00, image_dir_v06=None,
                 patient_ids=None, joint_types=['DIP'], img_size=128):
        self.img_size = img_size
        if image_dir_v06 is None:
            image_dir_v06 = image_dir_v00

        df = pd.read_csv(csv_path)
        df = df[df['joint_type'].isin(joint_types)].copy()

        if patient_ids is not None:
            df = df[df['patient_id'].isin(patient_ids)]

        df = df[df['v06_KL'] > df['v00_KL']].copy()

        df['v00_filename'] = df['patient_id'].astype(str) + '_' + df['joint'].str.lower() + '.png'
        df['v00_filepath'] = df['v00_filename'].apply(lambda f: os.path.join(image_dir_v00, f))

        df = df[df['v00_filepath'].apply(os.path.exists)].reset_index(drop=True)

        self.data = df
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_v00 = Image.open(row['v00_filepath']).convert('L')
        img_v00 = self.transform(img_v00)
        return {
            'image_v00': img_v00,
            'kl_v00': int(row['v00_KL']),
            'kl_v06': int(row['v06_KL']),
            'filename': row['v00_filename']
        }


# ============================================================
# Convenience: create_dataloaders
# ============================================================

def create_dataloaders(csv_path, image_dir, split_path=None,
                       joint_types=['DIP'], img_size=128,
                       batch_size=64, num_workers=4,
                       train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
                       seed=42):
    """
    Create train/val/test dataloaders with proper stratified split.

    If split_path exists, loads the saved split.
    Otherwise, creates a new split and saves it.
    """
    if split_path is None:
        split_dir = os.path.dirname(csv_path) or '.'
        split_path = os.path.join(
            split_dir, f'split_{"_".join(joint_types)}_{seed}.json'
        )

    if os.path.exists(split_path):
        split = load_patient_split(split_path)
    else:
        split = create_patient_split(
            csv_path, joint_types=joint_types,
            train_ratio=train_ratio, val_ratio=val_ratio,
            test_ratio=test_ratio, seed=seed,
            save_path=split_path
        )

    train_dataset = HandJointDataset(
        csv_path, image_dir,
        patient_ids=split['train'],
        joint_types=joint_types,
        img_size=img_size, augment=True
    )
    val_dataset = HandJointDataset(
        csv_path, image_dir,
        patient_ids=split['val'],
        joint_types=joint_types,
        img_size=img_size, augment=False
    )
    test_dataset = HandJointDataset(
        csv_path, image_dir,
        patient_ids=split['test'],
        joint_types=joint_types,
        img_size=img_size, augment=False
    )

    print(f"Datasets — Train: {len(train_dataset)} | "
          f"Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ============================================================
# Main: quick test / generate split
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='data/hand_long_clean2.csv')
    parser.add_argument('--joint_types', nargs='+', default=['DIP'])
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    get_kl_distribution(args.csv_path, args.joint_types, 'v00')
    get_kl_distribution(args.csv_path, args.joint_types, 'v06')

    split = create_patient_split(
        args.csv_path,
        joint_types=args.joint_types,
        train_ratio=0.70, val_ratio=0.15, test_ratio=0.15,
        seed=args.seed,
        save_path=f'data/split_{"_".join(args.joint_types)}_{args.seed}.json'
    )
