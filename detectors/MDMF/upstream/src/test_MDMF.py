"""
Test trained MDMF detector on test datasets.

This script evaluates a trained MDMF detector using standard MMD² scoring between
test samples and reference distribution.
"""

import os
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import pickle
import math
import matplotlib.pyplot as plt
from sklearn import metrics


# ================================
# Model Definition
# ================================

class MMD_MIL_Detector(nn.Module):
    """MMD-based Multiple Instance Learning detector."""
    
    def __init__(self, embed_dim=1024, hidden_dim=256, dropout=0.3, init_sigma=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.log_sigma = nn.Parameter(
            torch.log(torch.tensor([init_sigma])),
            requires_grad=True
        )
        
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def get_sigma(self):
        return torch.exp(self.log_sigma.clamp(min=-3.0, max=3.0))
    
    def compute_logits(self, embeddings, mask=None):
        B, K, D = embeddings.shape
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        flat = embeddings.view(B * K, D)
        logits = self.score_head(flat).view(B, K)
        
        if mask is not None:
            logits = logits.masked_fill(~mask, 0.0)
        
        return logits


# ================================
# DINOv2 Helper Functions
# ================================

def get_patch_tokens(backbone, x):
    """Extract patch tokens from DINOv2 backbone."""
    if hasattr(backbone, "forward_features"):
        out = backbone.forward_features(x)
    else:
        out = backbone(x)

    if isinstance(out, dict):
        tokens = None
        for key in ["x_norm_patchtokens", "patch_tokens", "tokens"]:
            if key in out:
                tokens = out[key]
                break
        
        if tokens is None:
            for key in ["x_norm", "x"]:
                if key in out:
                    t = out[key]
                    if t.dim() == 3 and t.size(1) > 1:
                        tokens = t[:, 1:, :]
                        break
        
        if tokens is None:
            raise RuntimeError(f"Cannot find patch tokens in output")
    else:
        if isinstance(out, (list, tuple)):
            out = out[0]
        
        if not torch.is_tensor(out):
            raise RuntimeError(f"Unsupported backbone output type: {type(out)}")
        
        if out.dim() == 3 and out.size(1) > 1:
            tokens = out[:, 1:, :]
        else:
            raise RuntimeError(f"Cannot interpret output shape {out.shape}")

    return tokens


def pool_tokens_to_patch_size(tokens, patch_size, image_size=224):
    """Pool token grid to match patch size."""
    B, N, D = tokens.shape
    H_t = int(math.sqrt(N))
    if H_t * H_t != N:
        raise ValueError(f"Token count {N} is not a perfect square")

    tokens_spatial = tokens.view(B, H_t, H_t, D)
    grid_size = image_size // patch_size
    if grid_size < 1:
        raise ValueError(f"Patch size {patch_size} too large for image size {image_size}")

    tokens_spatial_ch = tokens_spatial.permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(tokens_spatial_ch, output_size=(grid_size, grid_size))
    pooled = pooled.permute(0, 2, 3, 1).contiguous()
    pooled = pooled.view(B, grid_size * grid_size, D)

    return pooled


@torch.no_grad()
def extract_patch_logits_batch(images, backbone, model, patch_size):
    """
    Extract patch embeddings and compute logits for batch of images.
    
    Args:
        images: [B, C, H, W] on GPU
        backbone: DINOv2 backbone
        model: MMD-MIL detector
        patch_size: Virtual patch size
        
    Returns:
        patch_logits: [B, K] patch logits
    """
    with torch.cuda.amp.autocast(dtype=torch.float16, enabled=(images.dtype == torch.float16)):
        tokens = get_patch_tokens(backbone, images)
    
    tokens = tokens.float()
    patch_embeddings = pool_tokens_to_patch_size(tokens, patch_size=patch_size, image_size=224)
    patch_logits = model.compute_logits(patch_embeddings)
    
    return patch_logits


# ================================
# MMD² Scoring Functions
# ================================

@torch.no_grad()
def precompute_ref_stats(ref_X, sigma, epsilon=1e-8):
    """
    Precompute reference statistics for MMD² scoring.
    
    Args:
        ref_X: [n, d] reference samples (on GPU)
        sigma: Kernel bandwidth (on GPU)
        
    Returns:
        xx_mean: Mean of K(x, x') for reference
        ref_norm2: [n, 1] squared norms
    """
    n = ref_X.shape[0]
    sigma = sigma + epsilon
    
    ref_norm2 = (ref_X ** 2).sum(dim=1, keepdim=True)
    
    # Compute reference kernel matrix
    XX = ref_norm2
    distances = XX + XX.t() - 2.0 * (ref_X @ ref_X.t())
    Kxx = torch.exp(-distances / (2.0 * sigma**2))
    
    # Biased estimator (includes diagonal)
    xx_mean = Kxx.mean()
    
    return xx_mean, ref_norm2


@torch.no_grad()
def score_mmd2_batch(ref_X, ref_norm2, xx_mean, Y, sigma, epsilon=1e-8):
    """
    Compute MMD² scores for batch of test samples.
    
    MMD²(ref, {y}) = E[k(x,x')] + k(y,y) - 2*E[k(x,y)]
                   = xx_mean + 1 - 2*mean(k(x,y))
    
    Args:
        ref_X: [n, d] reference samples (on GPU)
        ref_norm2: [n, 1] reference squared norms (on GPU)
        xx_mean: Scalar, mean of reference kernel
        Y: [B, d] batch of test samples (on GPU)
        sigma: Kernel bandwidth (on GPU)
        
    Returns:
        scores: [B] MMD² scores (higher = more likely fake)
    """
    sigma = sigma + epsilon
    
    # Compute distances from test to reference
    Y_norm2 = (Y ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    distances = ref_norm2 + Y_norm2.t() - 2.0 * (ref_X @ Y.t())  # [n, B]
    
    # Compute kernel values
    kxy = torch.exp(-distances / (2.0 * sigma**2))  # [n, B]
    
    # Mean similarity for each test sample
    mean_kxy = kxy.mean(dim=0)  # [B]
    
    # MMD² = xx_mean + 1 - 2*mean_kxy
    mmd2_scores = xx_mean + 1.0 - 2.0 * mean_kxy
    
    return mmd2_scores


# ================================
# Load Reference Logits
# ================================

@torch.no_grad()
def load_reference_logits(ref_embeddings_path, model, device):
    """
    Load reference patch logits from pre-computed embeddings.
    
    Args:
        ref_embeddings_path: Path to reference embeddings (.pkl)
        model: MMD-MIL detector
        device: torch.device
        
    Returns:
        ref_logits: [N, K] reference logits (on GPU)
    """
    print("\n" + "="*70)
    print("Loading Reference Logits")
    print("="*70)
    
    with open(ref_embeddings_path, 'rb') as f:
        data = pickle.load(f)
    
    ref_embeddings = data['embeddings']
    ref_labels = data['labels']
    
    print(f"Reference dataset size: {len(ref_embeddings)} images")
    print(f"Patch size: {data['patch_size']}x{data['patch_size']}")
    
    # Only use real images (label=0)
    real_indices = [i for i, l in enumerate(ref_labels) if l == 0]
    ref_embeddings = [ref_embeddings[i] for i in real_indices]
    
    print(f"Using {len(ref_embeddings)} real images as reference")
    
    all_logits = []
    
    for emb in tqdm(ref_embeddings, desc="Computing reference logits"):
        emb_tensor = torch.from_numpy(emb).float().unsqueeze(0).to(device)
        patch_logits = model.compute_logits(emb_tensor)
        all_logits.append(patch_logits.squeeze(0))
    
    ref_logits = torch.stack(all_logits, dim=0)
    print(f"Reference logits shape: {ref_logits.shape}")
    print("="*70 + "\n")
    
    return ref_logits


# ================================
# Metrics and Plotting
# ================================

def calculate_metrics_and_plot(real_scores, fake_scores, dataset_name, output_dir):
    """Calculate metrics and generate plots."""
    
    all_scores = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
    
    # Calculate metrics
    fpr, tpr, _ = metrics.roc_curve(labels, all_scores)
    auroc = metrics.auc(fpr, tpr)
    
    tpr_threshold = 0.95
    valid_indices = tpr >= tpr_threshold
    fpr95 = fpr[valid_indices].min() if np.any(valid_indices) else 1.0
    
    ap = metrics.average_precision_score(labels, all_scores)
    
    # Print metrics
    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*70}")
    print(f"  Samples: Real={len(real_scores)}, Fake={len(fake_scores)}")
    print(f"  AUROC:             {auroc:.4f}")
    print(f"  FPR @ 95% TPR:     {fpr95:.4f}")
    print(f"  Average Precision: {ap:.4f}")
    print(f"\n  Real  - Score: {real_scores.mean():.6f} ± {real_scores.std():.6f}")
    print(f"  Fake  - Score: {fake_scores.mean():.6f} ± {fake_scores.std():.6f}")
    print(f"  Separation:    {fake_scores.mean() - real_scores.mean():.6f}")
    print(f"{'='*70}\n")
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Score distribution
    axes[0, 0].hist(real_scores, bins=50, alpha=0.7, density=True, 
                    color='#5B9BD5', label='Real', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(fake_scores, bins=50, alpha=0.7, density=True,
                    color='#A5665F', label='Fake', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('MMD² Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0, 0].set_title(f'Score Distribution: {dataset_name}', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Log-scale distribution
    axes[0, 1].hist(real_scores, bins=50, alpha=0.7, density=True,
                    color='#5B9BD5', label='Real', edgecolor='black', linewidth=0.5)
    axes[0, 1].hist(fake_scores, bins=50, alpha=0.7, density=True,
                    color='#A5665F', label='Fake', edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('MMD² Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Density (Log Scale)', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'Score Distribution (Log): {dataset_name}', fontsize=14, fontweight='bold')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # ROC curve
    axes[1, 0].plot(fpr, tpr, linewidth=2, color='#2E86AB', label=f'AUROC = {auroc:.4f}')
    axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[1, 0].scatter([fpr95], [0.95], color='red', s=100, zorder=5, 
                       label=f'FPR@95%TPR = {fpr95:.4f}')
    axes[1, 0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[1, 0].set_title(f'ROC Curve: {dataset_name}', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # PR curve
    precision, recall, _ = metrics.precision_recall_curve(labels, all_scores)
    axes[1, 1].plot(recall, precision, linewidth=2, color='#A23B72', label=f'AP = {ap:.4f}')
    axes[1, 1].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1, 1].set_title(f'Precision-Recall Curve: {dataset_name}', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, f'{dataset_name}_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'dataset': dataset_name,
        'num_real': int(len(real_scores)),
        'num_fake': int(len(fake_scores)),
        'auroc': float(auroc),
        'fpr_at_95_tpr': float(fpr95),
        'ap': float(ap),
        'real_score_mean': float(real_scores.mean()),
        'real_score_std': float(real_scores.std()),
        'fake_score_mean': float(fake_scores.mean()),
        'fake_score_std': float(fake_scores.std()),
        'score_difference': float(fake_scores.mean() - real_scores.mean())
    }


# ================================
# Main Testing Function
# ================================

def test_detector(model_path, ref_embeddings_path, test_real_dir, test_fake_dirs,
                  output_dir, batch_size=256):
    """
    Test trained MMD-MIL detector.
    
    Args:
        model_path: Path to trained model checkpoint
        ref_embeddings_path: Path to reference embeddings
        test_real_dir: Path to real test images
        test_fake_dirs: List of paths to fake test images
        output_dir: Output directory for results
        batch_size: Batch size for processing
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    patch_size = config.get('patch_size', 32)
    
    print(f"Model trained at epoch {checkpoint['epoch']}")
    print(f"Training AUROC: {checkpoint['auroc']:.4f}")
    
    model = MMD_MIL_Detector(
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        init_sigma=config['init_sigma']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    sigma = model.get_sigma()
    print(f"Learned kernel bandwidth (sigma): {sigma.item():.6f}")
    
    # Load DINOv2 backbone
    print("\nLoading DINOv2 backbone...")
    backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').to(device)
    backbone.eval()
    backbone.half()
    
    # Load reference logits
    ref_logits = load_reference_logits(ref_embeddings_path, model, device)
    
    # Precompute reference statistics
    print("Precomputing reference statistics...")
    xx_mean, ref_norm2 = precompute_ref_stats(ref_logits, sigma)
    print(f"Reference kernel mean: {xx_mean.item():.6f}")
    
    # Data transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*70)
    print("Starting Evaluation")
    print(f"Batch size: {batch_size}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"Scoring: MMD² (standard)")
    print("="*70)
    
    # Process real images
    print(f"\nProcessing real images from {test_real_dir}")
    real_dataset = datasets.ImageFolder(root=test_real_dir, transform=transform)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    real_scores_list = []
    
    for images, _ in tqdm(real_loader, desc="Testing real"):
        images = images.to(device, non_blocking=True).half()
        test_logits = extract_patch_logits_batch(images, backbone, model, patch_size)
        batch_scores = score_mmd2_batch(ref_logits, ref_norm2, xx_mean, test_logits, sigma)
        real_scores_list.append(batch_scores.cpu())
    
    real_scores = torch.cat(real_scores_list).numpy()
    
    print(f"Real images: {len(real_scores)}")
    print(f"  Score: {real_scores.mean():.6f} ± {real_scores.std():.6f}")
    
    # Process fake datasets
    all_results = []
    scores_dict = {'real': real_scores}
    
    for fake_dir in test_fake_dirs:
        dataset_name = Path(fake_dir).name
        print(f"\nProcessing fake dataset: {dataset_name}")
        
        fake_dataset = datasets.ImageFolder(root=fake_dir, transform=transform)
        fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=4, pin_memory=True)
        
        fake_scores_list = []
        
        for images, _ in tqdm(fake_loader, desc=f"Testing {dataset_name}"):
            images = images.to(device, non_blocking=True).half()
            test_logits = extract_patch_logits_batch(images, backbone, model, patch_size)
            batch_scores = score_mmd2_batch(ref_logits, ref_norm2, xx_mean, test_logits, sigma)
            fake_scores_list.append(batch_scores.cpu())
        
        fake_scores = torch.cat(fake_scores_list).numpy()
        scores_dict[dataset_name] = fake_scores
        
        print(f"Fake images: {len(fake_scores)}")
        print(f"  Score: {fake_scores.mean():.6f} ± {fake_scores.std():.6f}")
        
        # Calculate metrics
        result = calculate_metrics_and_plot(real_scores, fake_scores, dataset_name, plots_dir)
        all_results.append(result)
    
    # Calculate averages
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    avg_auroc = np.mean([r['auroc'] for r in all_results])
    avg_fpr95 = np.mean([r['fpr_at_95_tpr'] for r in all_results])
    avg_ap = np.mean([r['ap'] for r in all_results])
    
    print(f"\nAverage metrics across {len(all_results)} fake datasets:")
    print(f"  AUROC:             {avg_auroc:.4f}")
    print(f"  FPR @ 95% TPR:     {avg_fpr95:.4f}")
    print(f"  Average Precision: {avg_ap:.4f}")
    print("="*70)
    
    # Save results
    summary = {
        'model_path': model_path,
        'checkpoint_epoch': checkpoint['epoch'],
        'training_auroc': checkpoint['auroc'],
        'learned_sigma': float(sigma.item()),
        'patch_size': patch_size,
        'batch_size': batch_size,
        'num_ref_samples': int(ref_logits.shape[0]),
        'scoring_method': 'MMD² (standard)',
        'average_metrics': {
            'auroc': float(avg_auroc),
            'fpr_at_95_tpr': float(avg_fpr95),
            'ap': float(avg_ap)
        },
        'per_dataset_results': all_results,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    results_path = os.path.join(output_dir, 'test_results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"\nResults saved to {results_path}")
    
    pickle_path = os.path.join(output_dir, 'test_scores.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(scores_dict, f)
    print(f"Scores saved to {pickle_path}")
    
    return scores_dict, summary


# ================================
# Main
# ================================

def main():
    parser = argparse.ArgumentParser(
        description='Test MMD-MIL detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint (.pth)'
    )
    parser.add_argument(
        '--ref_embeddings',
        type=str,
        required=True,
        help='Path to reference embeddings (.pkl)'
    )
    parser.add_argument(
        '--test_real',
        type=str,
        required=True,
        help='Path to real test images directory'
    )
    parser.add_argument(
        '--test_fake',
        type=str,
        nargs='+',
        required=True,
        help='Paths to fake test images directories'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for results'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size for processing (default: 256)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("MMD-MIL Detector Testing")
    print("="*70)
    print(f"Model: {args.model}")
    print(f"Reference: {args.ref_embeddings}")
    print(f"Real test: {args.test_real}")
    print(f"Fake test: {len(args.test_fake)} datasets")
    for fake_dir in args.test_fake:
        print(f"  - {fake_dir}")
    print(f"Output: {args.output_dir}")
    print(f"Batch size: {args.batch_size}")
    print("="*70)
    
    scores_dict, summary = test_detector(
        model_path=args.model,
        ref_embeddings_path=args.ref_embeddings,
        test_real_dir=args.test_real,
        test_fake_dirs=args.test_fake,
        output_dir=args.output_dir,
        batch_size=args.batch_size
    )
    
    print("\n" + "="*70)
    print("TESTING COMPLETE!")
    print(f"Average AUROC: {summary['average_metrics']['auroc']:.4f}")
    print(f"Average FPR95: {summary['average_metrics']['fpr_at_95_tpr']:.4f}")
    print(f"Average AP:    {summary['average_metrics']['ap']:.4f}")
    print(f"Results saved to {args.output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()