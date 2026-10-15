"""
Pre-compute DINOv2 patch embeddings for training, validation, and reference sets.

This script extracts patch-level embeddings using DINOv2 ViT-L/14 backbone.
"""

import os
import math
import time
import random
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm


# ================================
# DINOv2 Helper Functions
# ================================

def get_patch_tokens(backbone, x):
    """
    Extract patch tokens from a DINOv2 backbone.
    
    Args:
        backbone: DINOv2 model
        x: Input tensor (B, C, H, W)
        
    Returns:
        tokens: Patch tokens (B, N, D) where N is number of patches
    """
    if hasattr(backbone, "forward_features"):
        out = backbone.forward_features(x)
    else:
        out = backbone(x)

    # Handle dict-like output
    if isinstance(out, dict):
        tokens = None
        
        # Try common keys for patch tokens
        for key in ["x_norm_patchtokens", "patch_tokens", "tokens"]:
            if key in out:
                tokens = out[key]
                break
        
        # Fallback: try full output with CLS token
        if tokens is None:
            for key in ["x_norm", "x"]:
                if key in out:
                    t = out[key]
                    if t.dim() == 3 and t.size(1) > 1:
                        tokens = t[:, 1:, :]  # Drop CLS token
                        break
        
        if tokens is None:
            raise RuntimeError(
                f"Cannot find patch tokens in output keys: {list(out.keys())}"
            )
    
    else:
        # Handle tensor or tuple/list output
        if isinstance(out, (list, tuple)):
            out = out[0]
        
        if not torch.is_tensor(out):
            raise RuntimeError(f"Unsupported backbone output type: {type(out)}")
        
        if out.dim() == 3 and out.size(1) > 1:
            # Assume format: [CLS] + patch tokens
            tokens = out[:, 1:, :]
        else:
            raise RuntimeError(
                f"Cannot interpret backbone output shape {out.shape} as patch tokens"
            )
    
    return tokens  # (B, N, D)


def pool_tokens_to_patch_size(tokens, patch_size, image_size=224):
    """
    Pool token grid to match desired patch grid size.
    
    Args:
        tokens: (B, N, D) patch tokens where N = H_t * W_t
        patch_size: Desired virtual patch size (e.g., 16, 32, 56)
        image_size: Input image size (default: 224)
        
    Returns:
        pooled: (B, K, D) where K = (image_size // patch_size)^2
    """
    B, N, D = tokens.shape
    H_t = int(math.sqrt(N))
    
    if H_t * H_t != N:
        raise ValueError(f"Token count {N} is not a perfect square")
    
    # Reshape to spatial grid: (B, H_t, W_t, D)
    tokens_spatial = tokens.view(B, H_t, H_t, D)
    
    # Calculate target grid size
    grid_size = image_size // patch_size
    if grid_size < 1:
        raise ValueError(
            f"Patch size {patch_size} is too large for image size {image_size}"
        )
    
    # Adaptive pooling: (B, D, H_t, W_t) -> (B, D, grid_size, grid_size)
    tokens_spatial_ch = tokens_spatial.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(
        tokens_spatial_ch,
        output_size=(grid_size, grid_size)
    )
    
    # Back to (B, K, D)
    pooled = pooled.permute(0, 2, 3, 1).contiguous()
    pooled = pooled.view(B, grid_size * grid_size, D)
    
    return pooled


# ================================
# Dataset Classes
# ================================

class TrainDataset(Dataset):
    """
    Training dataset with real/fake binary labels.
    
    Expected structure:
        data_root/
            class1/
                0_real/
                    img1.png
                    img2.png
                1_fake/
                    img1.png
                    img2.png
            class2/
                0_real/
                1_fake/
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        print("="*70)
        print("Scanning Training Dataset")
        print("="*70)
        
        # Collect all class directories
        class_dirs = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for cls_dir in class_dirs:
            cls_name = cls_dir.name
            print(f"\nProcessing class: {cls_name}")
            
            # Real images (label=0)
            real_dir = cls_dir / '0_real'
            if real_dir.exists():
                real_files = list(real_dir.glob('*.png')) + \
                            list(real_dir.glob('*.jpg')) + \
                            list(real_dir.glob('*.JPEG'))
                for img_path in real_files:
                    self.samples.append({
                        'path': str(img_path),
                        'label': 0,
                        'class': cls_name
                    })
                print(f"  Found {len(real_files)} real images")
            
            # Fake images (label=1)
            fake_dir = cls_dir / '1_fake'
            if fake_dir.exists():
                fake_files = list(fake_dir.glob('*.png')) + \
                            list(fake_dir.glob('*.jpg')) + \
                            list(fake_dir.glob('*.JPEG'))
                for img_path in fake_files:
                    self.samples.append({
                        'path': str(img_path),
                        'label': 1,
                        'class': cls_name
                    })
                print(f"  Found {len(fake_files)} fake images")
        
        print(f"\n{'='*70}")
        print(f"Total samples: {len(self.samples)}")
        print(f"  Real: {sum(1 for s in self.samples if s['label'] == 0)}")
        print(f"  Fake: {sum(1 for s in self.samples if s['label'] == 1)}")
        print('='*70)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample['class']


class ValDataset(Dataset):
    """
    Validation dataset with category-based organization.
    
    Expected structure:
        data_root/
            category1/
                0_real/
                    img1.png
                1_fake/
                    img1.png
            category2/
                0_real/
                1_fake/
    """
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
        
        print("="*70)
        print("Scanning Validation Dataset")
        print("="*70)
        
        # Collect all category directories
        categories = sorted([d for d in self.root_dir.iterdir() if d.is_dir()])
        
        for category_dir in categories:
            category = category_dir.name
            
            # Real images
            real_dir = category_dir / '0_real'
            if real_dir.exists():
                for img_path in real_dir.glob('*.png'):
                    self.samples.append({
                        'path': str(img_path),
                        'label': 0,
                        'category': category
                    })
            
            # Fake images
            fake_dir = category_dir / '1_fake'
            if fake_dir.exists():
                for img_path in fake_dir.glob('*.png'):
                    self.samples.append({
                        'path': str(img_path),
                        'label': 1,
                        'category': category
                    })
        
        print(f"Found {len(self.samples)} images")
        print(f"  Real: {sum(1 for s in self.samples if s['label'] == 0)}")
        print(f"  Fake: {sum(1 for s in self.samples if s['label'] == 1)}")
        print('='*70)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label'], sample.get('category', '')


# ================================
# Pre-compute Functions
# ================================

def precompute_train_val_embeddings(
    data_root,
    save_path,
    split='train',
    patch_size=32,
    batch_size=64
):
    """
    Pre-compute embeddings for training or validation set.
    
    Args:
        data_root: Path to dataset root
        save_path: Path to save embeddings (.pkl)
        split: 'train' or 'val'
        patch_size: Virtual patch size (e.g., 16, 32, 56)
        batch_size: Batch size for processing
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load DINOv2 backbone
    print("\nLoading DINOv2 ViT-L/14 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    backbone.to(device)
    backbone.eval()
    backbone.half()  # Use FP16 for efficiency
    print(f"Backbone loaded on {device}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    if split == 'train':
        dataset = TrainDataset(data_root, transform=transform)
    elif split == 'val':
        dataset = ValDataset(data_root, transform=transform)
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # DataLoader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Storage
    all_embeddings = []
    all_labels = []
    all_metadata = []
    embed_dim = None
    
    print(f"\nExtracting embeddings for {split} set...")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Batch size: {batch_size}")
    
    start_time = time.time()
    
    with torch.no_grad():
        for batch_images, batch_labels, batch_meta in tqdm(
            loader,
            desc=f"Processing {split}"
        ):
            B = batch_images.size(0)
            batch_images = batch_images.to(device, non_blocking=True).half()
            
            # Extract patch tokens
            with torch.cuda.amp.autocast(dtype=torch.float16):
                tokens = get_patch_tokens(backbone, batch_images)
            
            tokens = tokens.float()
            
            if embed_dim is None:
                embed_dim = tokens.shape[-1]
            
            # Pool to desired patch size
            pooled = pool_tokens_to_patch_size(
                tokens,
                patch_size=patch_size,
                image_size=224
            )
            pooled_np = pooled.cpu().numpy()
            
            # Store per-image
            for i in range(B):
                all_embeddings.append(pooled_np[i])  # (K, D)
                all_labels.append(int(batch_labels[i]))
                all_metadata.append(batch_meta[i] if isinstance(batch_meta, list) else '')
    
    elapsed = time.time() - start_time
    
    # Save to disk
    print(f"\nSaving embeddings to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        'embeddings': all_embeddings,
        'labels': all_labels,
        'metadata': all_metadata,
        'patch_size': patch_size,
        'split': split,
        'dataset_size': len(all_embeddings),
        'image_shape': (224, 224),
        'backbone': 'dinov2_vitl14',
        'embed_dim': embed_dim
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    # Print statistics
    print(f"\n{'='*70}")
    print(f"Embedding Extraction Complete ({split.upper()})")
    print(f"{'='*70}")
    print(f"Total images: {len(all_embeddings)}")
    print(f"Total patches: {sum(emb.shape[0] for emb in all_embeddings)}")
    print(f"Avg patches/image: {sum(emb.shape[0] for emb in all_embeddings) / len(all_embeddings):.1f}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Real samples: {sum(1 for l in all_labels if l == 0)}")
    print(f"Fake samples: {sum(1 for l in all_labels if l == 1)}")
    print(f"Time elapsed: {elapsed/60:.1f} min")
    print(f"File size: {os.path.getsize(save_path) / (1024**2):.1f} MB")
    print(f"Saved to: {save_path}")
    print('='*70)


def precompute_ref_embeddings(
    data_root,
    save_path,
    patch_size=32,
    images_per_class=5,
    seed=42
):
    """
    Pre-compute reference embeddings from ImageNet-style dataset.
    Randomly selects N images per class.
    
    Args:
        data_root: Path to ImageNet train folder
        save_path: Path to save embeddings (.pkl)
        patch_size: Virtual patch size (e.g., 16, 32, 56)
        images_per_class: Number of images to select per class
        seed: Random seed for reproducibility
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load backbone
    print("\nLoading DINOv2 ViT-L/14 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    backbone.to(device)
    backbone.eval()
    backbone.half()
    print(f"Backbone loaded on {device}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Collect class directories
    data_root = Path(data_root)
    class_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])
    
    if len(class_dirs) == 0:
        raise RuntimeError(f"No class directories found in {data_root}")
    
    print(f"\n{'='*70}")
    print("Reference Set Selection")
    print(f"{'='*70}")
    print(f"Found {len(class_dirs)} classes")
    print(f"Selecting {images_per_class} image(s) per class")
    print(f"Random seed: {seed}")
    
    # Select N images per class
    random.seed(seed)
    selected_images = []
    class_labels = []
    
    for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Selecting images")):
        images = list(class_dir.glob("*.png")) + \
                 list(class_dir.glob("*.jpg")) + \
                 list(class_dir.glob("*.JPEG"))
        
        if len(images) == 0:
            continue
        
        n_select = min(images_per_class, len(images))
        selected = random.sample(images, n_select)
        
        selected_images.extend(selected)
        class_labels.extend([class_idx] * n_select)
    
    print(f"Selected {len(selected_images)} images total")
    print(f"Average per class: {len(selected_images) / len(class_dirs):.1f}")
    
    # Extract embeddings
    all_embeddings = []
    image_paths = []
    image_class_labels = []
    embed_dim = None
    
    print("\nExtracting embeddings...")
    with torch.no_grad():
        for img_idx, img_path in enumerate(tqdm(selected_images, desc="Processing")):
            try:
                image = Image.open(img_path).convert('RGB')
                image = transform(image)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
            
            image = image.unsqueeze(0).to(device).half()
            
            # Extract tokens
            with torch.cuda.amp.autocast(dtype=torch.float16):
                tokens = get_patch_tokens(backbone, image)
            
            tokens = tokens.float()
            
            if embed_dim is None:
                embed_dim = tokens.shape[-1]
            
            # Pool to patch size
            patch_embeddings = pool_tokens_to_patch_size(
                tokens,
                patch_size=patch_size,
                image_size=224
            )
            
            patch_embeddings = patch_embeddings.squeeze(0).cpu().numpy()
            
            all_embeddings.append(patch_embeddings)
            image_paths.append(str(img_path))
            image_class_labels.append(class_labels[img_idx])
    
    # Save to disk
    print(f"\nSaving embeddings to {save_path}...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        'embeddings': all_embeddings,
        'labels': [0] * len(all_embeddings),  # All real images
        'class_labels': image_class_labels,
        'patch_size': patch_size,
        'split': 'ref',
        'dataset_size': len(all_embeddings),
        'image_paths': image_paths,
        'image_shape': (224, 224),
        'backbone': 'dinov2_vitl14',
        'embed_dim': embed_dim,
        'num_classes': len(class_dirs),
        'images_per_class': images_per_class,
        'seed': seed
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f, protocol=4)
    
    # Print statistics
    print(f"\n{'='*70}")
    print("Reference Embedding Extraction Complete")
    print(f"{'='*70}")
    print(f"Total images: {len(all_embeddings)}")
    print(f"Number of classes: {len(class_dirs)}")
    print(f"Images per class: {images_per_class}")
    print(f"Total patches: {sum(emb.shape[0] for emb in all_embeddings)}")
    print(f"Avg patches/image: {sum(emb.shape[0] for emb in all_embeddings) / len(all_embeddings):.1f}")
    print(f"Embedding dimension: {embed_dim}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"File size: {os.path.getsize(save_path) / (1024**2):.1f} MB")
    print(f"Random seed: {seed}")
    print(f"Saved to: {save_path}")
    print('='*70)


# ================================
# Main
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='Pre-compute DINOv2 patch embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument(
        '--split',
        type=str,
        required=True,
        choices=['train', 'val', 'ref'],
        help='Dataset split to process'
    )
    parser.add_argument(
        '--data_root',
        type=str,
        required=True,
        help='Path to dataset root directory'
    )
    parser.add_argument(
        '--save_path',
        type=str,
        required=True,
        help='Path to save embeddings (.pkl file)'
    )
    
    # Optional parameters
    parser.add_argument(
        '--patch_size',
        type=int,
        default=32,
        help='Virtual patch size (default: 32)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='Batch size for train/val (default: 64)'
    )
    parser.add_argument(
        '--images_per_class',
        type=int,
        default=5,
        help='Images per class for ref split (default: 5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for ref split (default: 42)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("DINOv2 Patch Embedding Pre-computation")
    print("="*70)
    print(f"Split: {args.split}")
    print(f"Data root: {args.data_root}")
    print(f"Save path: {args.save_path}")
    print(f"Patch size: {args.patch_size}")
    
    if args.split in ['train', 'val']:
        print(f"Batch size: {args.batch_size}")
        precompute_train_val_embeddings(
            data_root=args.data_root,
            save_path=args.save_path,
            split=args.split,
            patch_size=args.patch_size,
            batch_size=args.batch_size
        )
    
    elif args.split == 'ref':
        print(f"Images per class: {args.images_per_class}")
        print(f"Random seed: {args.seed}")
        precompute_ref_embeddings(
            data_root=args.data_root,
            save_path=args.save_path,
            patch_size=args.patch_size,
            images_per_class=args.images_per_class,
            seed=args.seed
        )


if __name__ == "__main__":
    main()