"""
Step 4: Train KL grade classifiers and compare:
  1. Baseline:     Real data only
  2. Traditional:  Real data + traditional augmentation (rotation, flip, etc.)
  3. LDM:          Real data + LDM-generated KL3/4 images
  4. Combined:     Real data + traditional aug + LDM-generated images

Usage:
    # Run all 4 experiments for DIP joint:
    python train_classifier.py --joint_types DIP --gen_dir generated_v2/ --experiment all

    # Run all 4 experiments for PIP joint:
    python train_classifier.py --joint_types PIP --gen_dir generated_pip/ --experiment all

    # Run all 4 experiments for MCP joint:
    python train_classifier.py --joint_types MCP --gen_dir generated_mcp/ --experiment all

    # Only run LDM experiment with specific noise strength:
    python train_classifier.py --joint_types DIP --gen_dir generated_v2/ --experiment ldm --guided_ns 0.5
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix,
                             balanced_accuracy_score, f1_score)
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import HandJointDataset, create_dataloaders, load_patient_split


# ============================================================
# Model
# ============================================================

class SimpleClassifier(nn.Module):
    """
    ResNet-18 classifier for KL grading (5 classes).
    Modified for 1-channel grayscale input.
    """
    def __init__(self, num_classes=5, pretrained=True):
        super().__init__()
        self.backbone = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )
        # Modify first conv for grayscale input (1 channel instead of 3)
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                         padding=3, bias=False)
        # Modify classifier head
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)


# ============================================================
# Generated Image Dataset
# ============================================================

class GeneratedImageDataset(Dataset):
    """Dataset for loading generated images from a directory."""
    def __init__(self, image_dir, kl_grade, img_size=128, augment=False):
        self.kl_grade = kl_grade
        self.image_paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.endswith('.png') and f.startswith('gen_')
        ])

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = self.transform(img)
        filename = os.path.basename(self.image_paths[idx])
        return {'image': img, 'kl_grade': self.kl_grade, 'filename': filename}


# ============================================================
# Training & Evaluation
# ============================================================

def train_classifier(model, train_loader, val_loader, device, epochs=30, lr=1e-4,
                     class_weights=None):
    """Train classifier and return best validation metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_bacc = 0
    best_metrics = {}
    best_state = None

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}', leave=False):
            images = batch['image'].to(device)
            labels = batch['kl_grade'].to(device)

            logits = model(images)
            loss = F.cross_entropy(logits, labels, weight=class_weights)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        # Validate
        val_metrics = evaluate_classifier(model, val_loader, device)

        if val_metrics['balanced_accuracy'] > best_val_bacc:
            best_val_bacc = val_metrics['balanced_accuracy']
            best_metrics = val_metrics.copy()
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3d}: train_acc={train_correct/max(train_total,1):.3f}, "
                  f"val_bacc={val_metrics['balanced_accuracy']:.3f}, "
                  f"val_f1_macro={val_metrics['f1_macro']:.3f}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_metrics


@torch.no_grad()
def evaluate_classifier(model, loader, device):
    """Evaluate classifier on given data."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['kl_grade']

        logits = model(images)
        preds = logits.argmax(1).cpu()

        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return {
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_per_class': f1_score(all_labels, all_preds, average=None, zero_division=0),
        'accuracy': (all_preds == all_labels).mean(),
        'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=range(5)),
        'predictions': all_preds,
        'labels': all_labels
    }


def extract_labels(dataset):
    """Extract KL labels from Dataset / ConcatDataset for class balancing."""
    if isinstance(dataset, HandJointDataset):
        return dataset.data['kl_grade'].astype(int).tolist()
    if isinstance(dataset, GeneratedImageDataset):
        return [int(dataset.kl_grade)] * len(dataset)
    if isinstance(dataset, ConcatDataset):
        labels = []
        for ds in dataset.datasets:
            labels.extend(extract_labels(ds))
        return labels
    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset)}")


# ============================================================
# Find generated image directories
# ============================================================

def find_generated_dirs(gen_dir, guided_ns=None, guided_only=True):
    """
    Find generated image subdirectories.

    Args:
        gen_dir: Root directory of generated images
        guided_ns: Which noise strength to use for guided images.
                   If None, picks the one with most images.
        guided_only: If True, exclude unconditional images (recommended).
    """
    found = []

    if not os.path.exists(gen_dir):
        print(f"  WARNING: gen_dir not found: {gen_dir}")
        return found

    for subdir in sorted(os.listdir(gen_dir)):
        full_path = os.path.join(gen_dir, subdir)
        if not os.path.isdir(full_path):
            continue

        # Count generated images
        gen_files = [f for f in os.listdir(full_path)
                     if f.endswith('.png') and f.startswith('gen_')]
        if len(gen_files) == 0:
            continue

        # Parse directory name
        if subdir.startswith('kl3_'):
            kl = 3
        elif subdir.startswith('kl4_'):
            kl = 4
        else:
            continue

        found.append({
            'path': full_path,
            'kl': kl,
            'name': subdir,
            'count': len(gen_files),
            'is_guided': 'guided' in subdir,
            'ns': None
        })

        # Extract noise strength if guided
        if 'guided_ns' in subdir:
            try:
                ns = float(subdir.split('ns')[1])
                found[-1]['ns'] = ns
            except (ValueError, IndexError):
                pass

    # Filter
    guided_dirs = [d for d in found if d['is_guided']]
    unconditional_dirs = [d for d in found if not d['is_guided']]

    selected = []
    if not guided_only:
        selected = list(unconditional_dirs)

    if guided_ns is not None:
        for d in guided_dirs:
            if d['ns'] is not None and abs(d['ns'] - guided_ns) < 0.01:
                selected.append(d)
    else:
        if guided_dirs:
            ns_counts = {}
            for d in guided_dirs:
                ns = d['ns']
                if ns is not None:
                    ns_counts[ns] = ns_counts.get(ns, 0) + d['count']
            if ns_counts:
                best_ns = max(ns_counts, key=ns_counts.get)
                for d in guided_dirs:
                    if d['ns'] is not None and abs(d['ns'] - best_ns) < 0.01:
                        selected.append(d)
                print(f"  Auto-selected guided noise_strength={best_ns}")
            else:
                selected.extend(guided_dirs)

    return selected


# ============================================================
# Plotting
# ============================================================

def plot_comparison(results, save_path):
    """Plot comparison of different experiments."""
    experiments = list(results.keys())
    metrics = ['balanced_accuracy', 'f1_macro', 'accuracy']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    for i, metric in enumerate(metrics):
        values = [results[exp][metric] for exp in experiments]
        bars = axes[i].bar(experiments, values,
                          color=colors[:len(experiments)])
        axes[i].set_title(metric.replace('_', ' ').title(), fontsize=12)
        axes[i].set_ylim(0, 1)
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{val:.3f}', ha='center', fontsize=10)
        axes[i].tick_params(axis='x', rotation=30)

    plt.suptitle('Classification Performance Comparison', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    # Per-class F1
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(5)
    width = 0.8 / len(experiments)

    for i, exp in enumerate(experiments):
        f1s = results[exp]['f1_per_class']
        bars = ax.bar(x + i * width, f1s, width, label=exp,
                     color=colors[i % len(colors)])

    ax.set_xlabel('KL Grade')
    ax.set_ylabel('F1 Score')
    ax.set_title('Per-Class F1 Score Comparison')
    ax.set_xticks(x + width * (len(experiments) - 1) / 2)
    ax.set_xticklabels([f'KL{i}' for i in range(5)])
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_path.replace('.png', '_per_class.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Train and compare KL grade classifiers')
    parser.add_argument('--data_dir', type=str, default='data/')
    parser.add_argument('--gen_dir', type=str, default='generated_v2/',
                        help='Directory with generated images')
    parser.add_argument('--joint_types', nargs='+', default=['DIP'])
    parser.add_argument('--guided_ns', type=float, default=None,
                        help='Which noise strength to use for guided images. '
                             'If not set, auto-picks the one with most images.')
    parser.add_argument('--guided_only', action='store_true', default=True,
                        help='Only use guided images, exclude unconditional (default: True)')
    parser.add_argument('--include_unconditional', action='store_true', default=False,
                        help='Also include unconditional generated images')
    parser.add_argument('--max_gen_ratio', type=float, default=0.5,
                        help='Max ratio of generated images to real KL3/4 images. '
                             'E.g., 0.5 means generated <= 50% of real. 0=no limit.')
    parser.add_argument('--k_values', nargs='+', type=float, default=None,
                        help='Optional sweep over K=generated/real for ldm/combined. '
                             'K should be in (0,1], e.g. --k_values 0.25 0.5')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--no_class_weights', action='store_true', default=False,
                        help='Disable inverse-frequency class weights in CE loss')
    parser.add_argument('--no_weighted_sampler', action='store_true', default=False,
                        help='Disable WeightedRandomSampler (use plain shuffle)')
    parser.add_argument('--drop_last', action='store_true', default=False,
                        help='Drop last incomplete batch (default: False)')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['baseline', 'traditional', 'ldm', 'combined', 'all'])
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='results/')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    joint_str = '_'.join(args.joint_types)
    output_dir = os.path.join(args.output_dir, joint_str)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Joint types: {args.joint_types}")
    print(f"Generated images dir: {args.gen_dir}")

    csv_path = os.path.join(args.data_dir, 'hand_long_clean2.csv')
    image_dir = os.path.join(args.data_dir, 'images')

    # ---- Create split (shared across all experiments) ----
    split_path = os.path.join(
        args.data_dir, f'split_{joint_str}_{args.seed}.json'
    )
    if os.path.exists(split_path):
        split = load_patient_split(split_path)
        print(f"Loaded existing split: {split_path}")
    else:
        from dataset import create_patient_split
        split = create_patient_split(
            csv_path, joint_types=args.joint_types, seed=args.seed,
            save_path=split_path
        )
        print(f"Created new split: {split_path}")

    print(f"Patients — Train: {len(split['train'])} | "
          f"Val: {len(split['val'])} | Test: {len(split['test'])}")

    # ---- Create val/test loaders (shared across all experiments) ----
    val_dataset = HandJointDataset(
        csv_path, image_dir,
        patient_ids=split['val'],
        joint_types=args.joint_types,
        img_size=args.img_size, augment=False
    )
    test_dataset = HandJointDataset(
        csv_path, image_dir,
        patient_ids=split['test'],
        joint_types=args.joint_types,
        img_size=args.img_size, augment=False
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                           shuffle=False, num_workers=args.num_workers,
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    print(f"Val images: {len(val_dataset)} | Test images: {len(test_dataset)}")

    # ---- Print KL distribution ----
    df = pd.read_csv(csv_path)
    df = df[df['joint_type'].isin(args.joint_types)]
    train_df = df[df['patient_id'].isin(split['train'])]
    print(f"\nTraining set KL distribution:")
    for kl in range(5):
        count = int((train_df['v00_KL'] == kl).sum())
        print(f"  KL{kl}: {count}")

    # ---- Find generated images ----
    guided_only = args.guided_only and (not args.include_unconditional)
    gen_info = find_generated_dirs(args.gen_dir, args.guided_ns, guided_only=guided_only)
    if gen_info:
        print(f"\nGenerated images found:")
        for d in gen_info:
            print(f"  {d['name']}: {d['count']} images (KL{d['kl']})")
    else:
        print(f"\nWARNING: No generated images found in {args.gen_dir}")

    # ---- Experiments ----
    experiments_to_run = (['baseline', 'traditional', 'ldm', 'combined']
                          if args.experiment == 'all'
                          else [args.experiment])

    k_sweep = None
    if args.k_values is not None:
        k_sweep = sorted(args.k_values)
        for k in k_sweep:
            if k <= 0 or k > 1:
                raise ValueError(
                    f"Invalid K in --k_values: {k}. For this codebase K=generated/real, "
                    "so K must be in (0, 1]."
                )
        print(f"K sweep enabled (K=generated/real): {k_sweep}")

    results = {}

    run_plan = []
    for exp_name in experiments_to_run:
        if exp_name in ['ldm', 'combined'] and k_sweep is not None:
            for k in k_sweep:
                run_plan.append({
                    'base_exp': exp_name,
                    'result_name': f'{exp_name}_K{k:g}',
                    'max_gen_ratio': k,
                    'k': k
                })
        else:
            run_plan.append({
                'base_exp': exp_name,
                'result_name': exp_name,
                'max_gen_ratio': args.max_gen_ratio,
                'k': None
            })

    for plan in run_plan:
        exp_name = plan['base_exp']
        result_name = plan['result_name']
        max_gen_ratio = plan['max_gen_ratio']

        print(f"\n{'='*60}")
        print(f"Experiment: {result_name}")
        if plan['k'] is not None:
            print(f"  K=generated/real={plan['k']:.4f} -> max_gen_ratio={max_gen_ratio:.4f}")
        print(f"{'='*60}")

        # Reset model for each experiment
        torch.manual_seed(args.seed)
        model = SimpleClassifier(num_classes=5, pretrained=True).to(device)

        # ---- Build training dataset ----
        if exp_name == 'baseline':
            # Real data only, NO augmentation
            train_dataset = HandJointDataset(
                csv_path, image_dir,
                patient_ids=split['train'],
                joint_types=args.joint_types,
                img_size=args.img_size, augment=False
            )
            print(f"  Training on {len(train_dataset)} real images (no augmentation)")

        elif exp_name == 'traditional':
            # Real data WITH traditional augmentation (flip, rotation, etc.)
            train_dataset = HandJointDataset(
                csv_path, image_dir,
                patient_ids=split['train'],
                joint_types=args.joint_types,
                img_size=args.img_size, augment=True
            )
            print(f"  Training on {len(train_dataset)} real images (with augmentation)")

        elif exp_name in ['ldm', 'combined']:
            # Real data + generated images
            augment = (exp_name == 'combined')
            real_dataset = HandJointDataset(
                csv_path, image_dir,
                patient_ids=split['train'],
                joint_types=args.joint_types,
                img_size=args.img_size, augment=augment
            )

            # Count real KL3/4 to cap generated images
            real_kl3 = int((train_df['v00_KL'] == 3).sum())
            real_kl4 = int((train_df['v00_KL'] == 4).sum())
            max_gen = {3: real_kl3, 4: real_kl4}
            if max_gen_ratio > 0:
                max_gen = {3: int(real_kl3 * max_gen_ratio),
                           4: int(real_kl4 * max_gen_ratio)}
                print(f"  Max generated: KL3<={max_gen[3]}, KL4<={max_gen[4]} "
                      f"(ratio={max_gen_ratio}x real)")

            gen_datasets = []
            total_gen = 0
            for d in gen_info:
                gen_ds = GeneratedImageDataset(
                    d['path'], d['kl'], args.img_size, augment=augment
                )
                # Cap number of generated images
                if max_gen_ratio > 0 and len(gen_ds) > max_gen[d['kl']]:
                    gen_ds.image_paths = gen_ds.image_paths[:max_gen[d['kl']]]
                    print(f"  + {d['name']}: capped to {len(gen_ds)} images (KL{d['kl']})")
                else:
                    print(f"  + {d['name']}: {len(gen_ds)} generated KL{d['kl']} images")
                gen_datasets.append(gen_ds)
                total_gen += len(gen_ds)

            if gen_datasets:
                train_dataset = ConcatDataset([real_dataset] + gen_datasets)
                print(f"  Training on {len(real_dataset)} real + {total_gen} generated "
                      f"= {len(train_dataset)} total"
                      f" ({'with' if augment else 'no'} augmentation)")
            else:
                train_dataset = real_dataset
                print(f"  WARNING: No generated images! Using real data only.")

        # ---- Class balancing ----
        train_labels = extract_labels(train_dataset)
        label_counts = np.bincount(train_labels, minlength=5)
        print(f"  Train label counts (after augmentation concat): {label_counts.tolist()}")

        class_weights = None
        if not args.no_class_weights:
            class_weights_np = np.zeros(5, dtype=np.float32)
            non_zero = label_counts > 0
            class_weights_np[non_zero] = (
                len(train_labels) / (5.0 * label_counts[non_zero])
            )
            class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=device)
            print(f"  CE class weights: {[round(float(w), 4) for w in class_weights]}")

        sampler = None
        if not args.no_weighted_sampler:
            sample_weights = [1.0 / max(label_counts[y], 1) for y in train_labels]
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(sample_weights),
                num_samples=len(sample_weights),
                replacement=True
            )
            print("  Using WeightedRandomSampler for balanced mini-batches")

        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=(sampler is None), sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True, drop_last=args.drop_last
        )

        # ---- Train ----
        val_metrics = train_classifier(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, class_weights=class_weights
        )

        # ---- Evaluate on TEST set ----
        test_metrics = evaluate_classifier(model, test_loader, device)
        print(f"\n  === Test Set Results for {result_name} ===")
        print(f"    Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
        print(f"    F1 Macro:          {test_metrics['f1_macro']:.4f}")
        print(f"    Accuracy:          {test_metrics['accuracy']:.4f}")
        print(f"    F1 per class:      {[f'{f:.3f}' for f in test_metrics['f1_per_class']]}")
        print(f"\n  Classification Report:")
        print(classification_report(
            test_metrics['labels'], test_metrics['predictions'],
            target_names=[f'KL{i}' for i in range(5)],
            zero_division=0
        ))

        results[result_name] = test_metrics

        # Save confusion matrix
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(test_metrics['confusion_matrix'], annot=True, fmt='d',
                    xticklabels=[f'KL{i}' for i in range(5)],
                    yticklabels=[f'KL{i}' for i in range(5)],
                    cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix - {exp_name} ({joint_str})')
        plt.tight_layout()
        safe_name = result_name.replace('.', 'p')
        plt.savefig(os.path.join(output_dir, f'confusion_{safe_name}.png'), dpi=150)
        plt.close()

        # Save model
        torch.save(model.state_dict(),
                   os.path.join(output_dir, f'classifier_{safe_name}.pt'))

    # ---- Summary ----
    if len(results) > 1:
        plot_comparison(results, os.path.join(output_dir, 'comparison.png'))

        print(f"\n{'='*80}")
        print(f"  SUMMARY — {joint_str} (Test Set)")
        print(f"{'='*80}")
        print(f"{'Experiment':<15} {'Bal.Acc':>8} {'F1-Macro':>9} {'Accuracy':>9} "
              f"{'F1-KL0':>7} {'F1-KL1':>7} {'F1-KL2':>7} {'F1-KL3':>7} {'F1-KL4':>7}")
        print(f"{'-'*80}")
        for exp, m in results.items():
            f1s = m['f1_per_class']
            f1_vals = [f1s[i] if i < len(f1s) else 0 for i in range(5)]
            print(f"{exp:<15} {m['balanced_accuracy']:>8.4f} {m['f1_macro']:>9.4f} "
                  f"{m['accuracy']:>9.4f} "
                  + " ".join(f"{v:>7.3f}" for v in f1_vals))

        # Save summary to JSON
        summary = {}
        for exp, m in results.items():
            summary[exp] = {
                'balanced_accuracy': float(m['balanced_accuracy']),
                'f1_macro': float(m['f1_macro']),
                'accuracy': float(m['accuracy']),
                'f1_per_class': [float(f) for f in m['f1_per_class']],
                'confusion_matrix': m['confusion_matrix'].tolist()
            }
        with open(os.path.join(output_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to {output_dir}/")


if __name__ == '__main__':
    main()
