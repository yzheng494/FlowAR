"""
Splits 100 images from the training set into a validation set.

- Symlinks raw images into data_path/val/ (used by evaluate_reconstruction)
- Removes the corresponding .npz files from cache_dir (excludes them from CachedFolder training)

Run once before training:
    python create_val_split.py
"""

import argparse
import os
import random
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_path',   default='/localscratch/yzheng494/FlowAR')
    p.add_argument('--cached_path', default='/localscratch/yzheng494/FlowAR/cache_dir')
    p.add_argument('--num_images',  type=int, default=100)
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    train_dir  = Path(args.data_path) / 'train'
    val_dir    = Path(args.data_path) / 'val'
    cache_dir  = Path(args.cached_path)

    if val_dir.exists() and any(val_dir.iterdir()):
        print(f"Val dir already exists and is non-empty: {val_dir}")
        print("Delete it first if you want to recreate the split.")
        return

    rng = random.Random(args.seed)

    # collect all training images with their cache counterpart
    candidates = []
    for cls_dir in sorted(train_dir.iterdir()):
        if not cls_dir.is_dir():
            continue
        for img in sorted(cls_dir.glob('*')):
            if img.suffix.lower() not in {'.jpeg', '.jpg', '.png'}:
                continue
            npz = cache_dir / cls_dir.name / (img.name + '.npz')
            candidates.append((img, npz, cls_dir.name))

    rng.shuffle(candidates)
    chosen = candidates[:args.num_images]

    removed_npz = 0
    created_links = 0
    missing_npz = []

    for img_path, npz_path, cls_name in chosen:
        # create symlink in val/
        out_cls = val_dir / cls_name
        out_cls.mkdir(parents=True, exist_ok=True)
        link = out_cls / img_path.name
        if not link.exists():
            link.symlink_to(img_path.resolve())
            created_links += 1

        # remove from cache so CachedFolder won't see it during training
        if npz_path.exists():
            npz_path.unlink()
            removed_npz += 1
        else:
            missing_npz.append(str(npz_path))

    print(f"Created {created_links} symlinks in {val_dir}")
    print(f"Removed {removed_npz} .npz files from cache")
    if missing_npz:
        print(f"Warning: {len(missing_npz)} .npz files not found in cache (already missing?):")
        for p in missing_npz[:5]:
            print(f"  {p}")


if __name__ == '__main__':
    main()
