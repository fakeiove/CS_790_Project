"""
Utility functions for:
- FID (Frechet Inception Distance) calculation
- Image quality metrics
- Visualization helpers
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from scipy import linalg
from tqdm import tqdm


class ImageFolderFlat(Dataset):
    """Simple dataset loading all PNG images from a directory."""
    def __init__(self, image_dir, img_size=128):
        self.paths = sorted([
            os.path.join(image_dir, f) for f in os.listdir(image_dir)
            if f.lower().endswith('.png')
        ])
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('L')
        return self.transform(img)


class InceptionFeatureExtractor(nn.Module):
    """
    Extract features from a pretrained InceptionV3 for FID calculation.
    Modified for grayscale input.
    """
    def __init__(self, device='cuda'):
        super().__init__()
        inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        inception.eval()

        # Modify for grayscale: replicate channel
        self.preprocess = nn.Conv2d(1, 3, 1, bias=False)
        nn.init.constant_(self.preprocess.weight, 1.0)

        # Use features up to average pool (2048-dim)
        self.features = nn.Sequential(
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Conv2d_3b_1x1,
            inception.Conv2d_4a_3x3,
            nn.MaxPool2d(3, stride=2),
            inception.Mixed_5b,
            inception.Mixed_5c,
            inception.Mixed_5d,
            inception.Mixed_6a,
            inception.Mixed_6b,
            inception.Mixed_6c,
            inception.Mixed_6d,
            inception.Mixed_6e,
            inception.Mixed_7a,
            inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x):
        # x: [B, 1, H, W], range [-1, 1]
        # Resize to 299x299 for Inception
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear')
        x = self.preprocess(x)
        features = self.features(x)
        return features.squeeze(-1).squeeze(-1)  # [B, 2048]


@torch.no_grad()
def compute_fid(real_loader, fake_loader, device='cuda', batch_size=32):
    """
    Compute FID between real and generated image distributions.
    Lower FID = generated images are closer to real distribution.
    """
    extractor = InceptionFeatureExtractor(device)
    extractor.eval()

    def get_features(loader):
        all_features = []
        for batch in tqdm(loader, desc='Extracting features', leave=False):
            if isinstance(batch, dict):
                imgs = batch['image'].to(device)
            else:
                imgs = batch.to(device)
            feats = extractor(imgs)
            all_features.append(feats.cpu().numpy())
        return np.concatenate(all_features, axis=0)

    print("Computing features for real images...")
    real_feats = get_features(real_loader)
    print("Computing features for generated images...")
    fake_feats = get_features(fake_loader)

    # Compute statistics
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    mu_fake = np.mean(fake_feats, axis=0)
    sigma_fake = np.cov(fake_feats, rowvar=False)

    # FID formula
    diff = mu_real - mu_fake
    covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)

    return float(fid)


def compute_fid_from_dirs(real_dir, fake_dir, device='cuda',
                          img_size=128, batch_size=32):
    """Compute FID between images in two directories."""
    real_dataset = ImageFolderFlat(real_dir, img_size)
    fake_dataset = ImageFolderFlat(fake_dir, img_size)

    real_loader = DataLoader(real_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=4)

    print(f"Real images: {len(real_dataset)}, Generated images: {len(fake_dataset)}")
    fid = compute_fid(real_loader, fake_loader, device)
    print(f"FID: {fid:.2f}")
    return fid


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_dir', type=str, required=True)
    parser.add_argument('--fake_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fid = compute_fid_from_dirs(args.real_dir, args.fake_dir, device, args.img_size)
    print(f"\nFID Score: {fid:.2f}")
