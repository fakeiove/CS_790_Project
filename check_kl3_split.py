"""统计划分数据集后，三种关节 KL=3 在 train/val/test 中各有多少张。"""
import os
import json
import pandas as pd
from dataset import load_patient_split, HandJointDataset

csv_path = 'data/hand_long_clean2.csv'
image_dir = 'data/images'
df_all = pd.read_csv(csv_path)

for jt in ['DIP', 'PIP', 'MCP']:
    print(f"\n{'='*50}")
    print(f"  {jt}")
    print(f"{'='*50}")

    split_file = f'data/split_{jt}_42.json'
    if not os.path.exists(split_file):
        print(f"  Split file not found: {split_file}")
        continue

    with open(split_file) as f:
        split = json.load(f)

    df = df_all[df_all['joint_type'] == jt].copy()

    for part in ['train', 'val', 'test']:
        pids = split[part]
        sub = df[df['patient_id'].isin(pids)]
        total = len(sub)
        kl3 = int((sub['v00_KL'] == 3).sum())
        kl4 = int((sub['v00_KL'] == 4).sum())
        print(f"  {part:>5s}: total={total:>5d}, KL3={kl3:>4d}, KL4={kl4:>4d}")
