"""
Train MDMF detector for AIGC detection.

This script trains a MDMF detector to distinguish between real and AI-generated images at the patch level.
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import pickle
from sklearn import metrics


# ================================
# Dataset
# ================================

class PrecomputedEmbeddingDataset(Dataset):
    """Dataset that loads pre-computed patch embeddings."""
    
    def __init__(self, embeddings_path, label=None):
        """
        Args:
            embeddings_path: Path to pre-computed embeddings (.pkl file)
            label: Optional label filter (0=real, 1=fake, None=all)
        """
        print(f"Loading embeddings from {embeddings_path}...")
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        all_embeddings = data['embeddings']  # List of (K, D) arrays
        all_labels = data['labels']
        self.patch_size = data['patch_size']
        
        if label is not None:
            # Filter by label
            indices = [i for i, l in enumerate(all_labels) if l == label]
            self.embeddings = [all_embeddings[i] for i in indices]
            self.labels = [all_labels[i] for i in indices]
            label_name = "Real" if label == 0 else "Fake"
            print(f"[{label_name}] Loaded {len(self.embeddings)} images")
        else:
            self.embeddings = all_embeddings
            self.labels = all_labels
            print(f"Loaded {len(self.embeddings)} images")
        
        print(f"Patch size: {self.patch_size}x{self.patch_size}")
        real_count = sum(1 for l in self.labels if l == 0)
        fake_count = sum(1 for l in self.labels if l == 1)
        print(f"Distribution: Real={real_count}, Fake={fake_count}")
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        embeddings = torch.from_numpy(self.embeddings[idx]).float()  # (K, D)
        label = self.labels[idx]
        return embeddings, label


def collate_fn(batch):
    """
    Custom collate function to handle variable-length patch sequences.
    
    Args:
        batch: List of (embeddings, label) tuples
        
    Returns:
        padded_embeddings: [B, K_max, D] padded tensor
        labels: [B] label tensor
        mask: [B, K_max] boolean mask (True for valid patches)
    """
    embeddings, labels = zip(*batch)
    max_patches = max(emb.shape[0] for emb in embeddings)
    B = len(embeddings)
    D = embeddings[0].shape[1]
    
    padded_embeddings = torch.zeros(B, max_patches, D)
    mask = torch.zeros(B, max_patches, dtype=torch.bool)
    
    for i, emb in enumerate(embeddings):
        K = emb.shape[0]
        padded_embeddings[i, :K] = emb
        mask[i, :K] = True
    
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_embeddings, labels, mask


# ================================
# Kernel and MMD Functions
# ================================

def rbf_kernel(X, Y, sigma):
    """
    Compute RBF (Gaussian) kernel matrix between X and Y.
    
    K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
    
    Args:
        X: [n, d] tensor
        Y: [m, d] tensor
        sigma: kernel bandwidth
        
    Returns:
        K: [n, m] kernel matrix
    """
    XX = torch.sum(X ** 2, dim=1, keepdim=True)  # [n, 1]
    YY = torch.sum(Y ** 2, dim=1, keepdim=True)  # [m, 1]
    distances = XX + YY.t() - 2 * (X @ Y.t())    # [n, m]
    return torch.exp(-distances / (2 * sigma ** 2))


def mmd_squared(X, Y, sigma, epsilon=1e-8):
    """
    Compute unbiased MMD^2 estimate between two samples.
    
    MMD^2 = E[k(x,x')] - 2*E[k(x,y)] + E[k(y,y')]
    
    Uses unbiased estimators (excludes diagonal terms).
    
    Args:
        X: [n, d] real sample
        Y: [m, d] fake sample
        sigma: kernel bandwidth
        epsilon: numerical stability constant
        
    Returns:
        mmd2: Scalar MMD^2 value
    """
    n = X.shape[0]
    m = Y.shape[0]
    assert X.shape[1] == Y.shape[1], f"Dimension mismatch: X={X.shape[1]}, Y={Y.shape[1]}"
    
    sigma = sigma + epsilon
    
    # Compute kernel matrices
    Kxx = rbf_kernel(X, X, sigma)
    Kyy = rbf_kernel(Y, Y, sigma)
    Kxy = rbf_kernel(X, Y, sigma)
    
    # Unbiased estimators (exclude diagonal)
    if n > 1:
        xx = (torch.sum(Kxx) - torch.sum(torch.diag(Kxx))) / (n * (n - 1))
    else:
        xx = torch.tensor(0.0, device=X.device)
    
    if m > 1:
        yy = (torch.sum(Kyy) - torch.sum(torch.diag(Kyy))) / (m * (m - 1))
    else:
        yy = torch.tensor(0.0, device=Y.device)
    
    xy = torch.sum(Kxy) / (n * m)
    
    return xx + yy - 2.0 * xy


# ================================
# Model
# ================================

class MMD_MIL_Detector(nn.Module):
    """
    MMD-based Multiple Instance Learning detector.
    
    Architecture:
        1. MLP head: maps patch embeddings to scalar logits
        2. Learnable kernel bandwidth (sigma)
        3. Training objective: maximize MMD^2 between real/fake distributions
    """
    
    def __init__(self, embed_dim=1024, hidden_dim=256, dropout=0.3, init_sigma=1.0):
        """
        Args:
            embed_dim: Input embedding dimension (e.g., 1024 for DINOv2-L)
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            init_sigma: Initial kernel bandwidth
        """
        super().__init__()
        self.embed_dim = embed_dim
        
        # Learnable kernel bandwidth (log-parameterized)
        self.log_sigma = nn.Parameter(
            torch.log(torch.tensor([init_sigma])),
            requires_grad=True
        )
        
        # MLP head: embedding -> logit
        self.score_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Bounded output in [-1, 1]
        )
        
        print(f"Initialized MMD-MIL Detector:")
        print(f"  Embed dim: {embed_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Dropout: {dropout}")
        print(f"  Initial sigma: {init_sigma}")
    
    def get_sigma(self):
        """Get current kernel bandwidth with range constraint."""
        # Constrain sigma to [exp(-3), exp(3)] ≈ [0.05, 20]
        return torch.exp(self.log_sigma.clamp(min=-3.0, max=3.0))
    
    def compute_logits(self, embeddings, mask=None):
        """
        Compute patch-level logits.
        
        Args:
            embeddings: [B, K, D] patch embeddings
            mask: [B, K] boolean mask for valid patches
            
        Returns:
            logits: [B, K] patch logits
        """
        B, K, D = embeddings.shape
        
        # L2 normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
        # Flatten and pass through MLP
        flat = embeddings.view(B * K, D)
        logits = self.score_head(flat).view(B, K)
        
        # Mask out invalid patches
        if mask is not None:
            logits = logits.masked_fill(~mask, 0.0)
        
        return logits
    
    def forward(self, real_embeddings, fake_embeddings, real_mask=None, fake_mask=None):
        """
        Forward pass for training.
        
        Args:
            real_embeddings: [B, K, D] real image embeddings
            fake_embeddings: [B, K, D] fake image embeddings
            real_mask: [B, K] mask for real patches
            fake_mask: [B, K] mask for fake patches
            
        Returns:
            loss: Training loss (-MMD^2)
            mmd2: MMD^2 value
            sigma: Current kernel bandwidth
        """
        # Compute logits for both distributions
        real_logits = self.compute_logits(real_embeddings, real_mask)  # [B, K]
        fake_logits = self.compute_logits(fake_embeddings, fake_mask)  # [B, K]
        
        # Flatten to get distributions
        X = real_logits.view(-1, 1)  # [B*K, 1] - treat each patch as sample
        Y = fake_logits.view(-1, 1)  # [B*K, 1]
        
        sigma = self.get_sigma()
        
        # Compute MMD^2
        mmd2 = mmd_squared(X, Y, sigma)
        
        # Loss: maximize MMD^2 => minimize -MMD^2
        loss = -mmd2
        
        return loss, mmd2, sigma


# ================================
# Training Functions
# ================================

def train_epoch(model, train_loaders, optimizer, device, epoch):
    """
    Train for one epoch.
    
    Args:
        model: MMD-MIL detector
        train_loaders: Dict with 'real' and 'fake' DataLoaders
        optimizer: Optimizer
        device: torch.device
        epoch: Current epoch number
        
    Returns:
        avg_loss: Average loss
        avg_mmd: Average MMD^2
    """
    model.train()
    
    real_loader = train_loaders["real"]
    fake_loader = train_loaders["fake"]
    
    total_loss = 0.0
    total_mmd = 0.0
    num_batches = 0
    
    pbar = tqdm(
        zip(real_loader, fake_loader),
        desc=f"Epoch {epoch}",
        total=min(len(real_loader), len(fake_loader))
    )
    
    for (real_emb, _, real_mask), (fake_emb, _, fake_mask) in pbar:
        # Skip if batch sizes don't match
        if real_emb.shape[0] != fake_emb.shape[0]:
            continue
        
        # Move to device
        real_emb = real_emb.to(device)
        fake_emb = fake_emb.to(device)
        real_mask = real_mask.to(device)
        fake_mask = fake_mask.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        loss, mmd2, sigma = model(real_emb, fake_emb, real_mask, fake_mask)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Accumulate stats
        total_loss += loss.item()
        total_mmd += mmd2.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "mmd2": f"{mmd2.item():.6f}",
            "sigma": f"{sigma.item():.4f}"
        })
    
    avg_loss = total_loss / max(1, num_batches)
    avg_mmd = total_mmd / max(1, num_batches)
    
    return avg_loss, avg_mmd


# ================================
# Validation Functions
# ================================

@torch.no_grad()
def extract_ref_logits(model, ref_loader, device):
    """
    Extract logits for reference set (all real images).
    
    Args:
        model: MMD-MIL detector
        ref_loader: Reference set DataLoader
        device: torch.device
        
    Returns:
        ref_logits: [N, K] tensor of reference logits
    """
    model.eval()
    all_logits = []
    
    for embeddings, _, mask in tqdm(ref_loader, desc="Extracting ref logits"):
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        logits = model.compute_logits(embeddings, mask)
        all_logits.append(logits.cpu())
    
    return torch.cat(all_logits, dim=0)  # [N, K]


@torch.no_grad()
def precompute_ref_stats(ref_logits, sigma, epsilon=1e-8):
    """
    Precompute reference statistics for fast scoring.
    
    Args:
        ref_logits: [N, K] reference logits
        sigma: Kernel bandwidth
        epsilon: Numerical stability
        
    Returns:
        xx_mean: Mean of K(x, x') for reference
        ref_norm2: [N, K] squared norms for distance computation
    """
    sigma = sigma + epsilon
    N, K = ref_logits.shape
    
    # Compute squared norms
    ref_norm2 = (ref_logits ** 2).sum(dim=1, keepdim=True)  # [N, 1]
    
    # Compute reference kernel matrix
    XX = ref_norm2
    distances = XX + XX.t() - 2.0 * (ref_logits @ ref_logits.t())
    Kxx = torch.exp(-distances / (2.0 * sigma ** 2))
    
    # Mean kernel value (biased estimator, includes diagonal)
    xx_mean = Kxx.mean()
    
    return xx_mean, ref_norm2


@torch.no_grad()
def score_sample(test_logit, ref_logits, ref_norm2, xx_mean, sigma, epsilon=1e-8):
    """
    Score a single test sample using MMD^2(ref, {test}).
    
    For a single sample y:
        MMD^2(ref, {y}) = E[k(x,x')] + k(y,y) - 2*E[k(x,y)]
                        = xx_mean + 1 - 2*mean(k(x,y))
    
    Args:
        test_logit: [1, K] test sample logits
        ref_logits: [N, K] reference logits
        ref_norm2: [N, 1] reference squared norms
        xx_mean: Scalar, mean of reference kernel
        sigma: Kernel bandwidth
        epsilon: Numerical stability
        
    Returns:
        score: Scalar MMD^2 score (higher = more likely fake)
    """
    sigma = sigma + epsilon
    
    # Compute distances from test to all reference samples
    y_norm2 = (test_logit ** 2).sum(dim=1, keepdim=True)  # [1, 1]
    distances = ref_norm2 + y_norm2 - 2.0 * (ref_logits @ test_logit.t())  # [N, 1]
    
    # Compute kernel values
    kxy = torch.exp(-distances / (2.0 * sigma ** 2))  # [N, 1]
    
    # Mean similarity
    mean_kxy = kxy.mean()
    
    # MMD^2 = xx_mean + 1 - 2*mean_kxy
    mmd2 = xx_mean + 1.0 - 2.0 * mean_kxy
    
    return mmd2.item()


@torch.no_grad()
def validate(model, val_loaders, ref_loader, device):
    """
    Validate model on validation set.
    
    Args:
        model: MMD-MIL detector
        val_loaders: Dict with 'real' and 'fake' DataLoaders
        ref_loader: Reference set DataLoader
        device: torch.device
        
    Returns:
        auroc: Area under ROC curve
        fpr95: False positive rate at 95% TPR
    """
    model.eval()
    sigma = model.get_sigma()
    
    print(f"\nValidation - Kernel bandwidth: {sigma.item():.6f}")
    
    # Extract and precompute reference statistics
    print("Extracting reference logits...")
    ref_logits = extract_ref_logits(model, ref_loader, device).to(device)
    print(f"Reference shape: {ref_logits.shape}")
    
    print("Precomputing reference statistics...")
    xx_mean, ref_norm2 = precompute_ref_stats(ref_logits, sigma)
    print(f"Reference kernel mean: {xx_mean.item():.6f}")
    
    # Score validation samples
    all_scores = []
    all_labels = []
    
    # Score real samples
    print("Scoring real samples...")
    for embeddings, _, mask in tqdm(val_loaders["real"], desc="Real"):
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        test_logits = model.compute_logits(embeddings, mask)  # [B, K]
        
        for i in range(test_logits.shape[0]):
            y = test_logits[i:i+1]
            score = score_sample(y, ref_logits, ref_norm2, xx_mean, sigma)
            
            if np.isfinite(score):
                all_scores.append(score)
                all_labels.append(0)
    
    # Score fake samples
    print("Scoring fake samples...")
    for embeddings, _, mask in tqdm(val_loaders["fake"], desc="Fake"):
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        test_logits = model.compute_logits(embeddings, mask)  # [B, K]
        
        for i in range(test_logits.shape[0]):
            y = test_logits[i:i+1]
            score = score_sample(y, ref_logits, ref_norm2, xx_mean, sigma)
            
            if np.isfinite(score):
                all_scores.append(score)
                all_labels.append(1)
    
    # Convert to numpy
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    fpr, tpr, _ = metrics.roc_curve(all_labels, all_scores)
    auroc = metrics.auc(fpr, tpr)
    
    # FPR at 95% TPR
    tpr_threshold = 0.95
    valid_indices = tpr >= tpr_threshold
    fpr95 = fpr[valid_indices].min() if np.any(valid_indices) else 1.0
    
    # Print statistics
    real_scores = all_scores[all_labels == 0]
    fake_scores = all_scores[all_labels == 1]
    
    print(f"\nScore Statistics:")
    print(f"  Real: mean={real_scores.mean():.6f}, std={real_scores.std():.6f}")
    print(f"  Fake: mean={fake_scores.mean():.6f}, std={fake_scores.std():.6f}")
    print(f"  Separation: {fake_scores.mean() - real_scores.mean():.6f}")
    
    return auroc, fpr95


# ================================
# Main Training Loop
# ================================

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train MMD-MIL detector',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config JSON file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for checkpoints and logs'
    )
    
    args = parser.parse_args()
    
    # Load config
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to output directory
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("\n" + "="*70)
    print("Training Configuration")
    print("="*70)
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("="*70)
    
    # Load datasets
    print("\n" + "="*70)
    print("Loading Datasets")
    print("="*70)
    
    train_datasets = {
        "real": PrecomputedEmbeddingDataset(config['train_embeddings'], label=0),
        "fake": PrecomputedEmbeddingDataset(config['train_embeddings'], label=1)
    }
    
    val_datasets = {
        "real": PrecomputedEmbeddingDataset(config['val_embeddings'], label=0),
        "fake": PrecomputedEmbeddingDataset(config['val_embeddings'], label=1)
    }
    
    ref_dataset = PrecomputedEmbeddingDataset(config['ref_embeddings'], label=0)
    
    # Create data loaders
    train_loaders = {
        "real": DataLoader(
            train_datasets["real"],
            batch_size=config['batch_size'] // 2,
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        ),
        "fake": DataLoader(
            train_datasets["fake"],
            batch_size=config['batch_size'] // 2,
            shuffle=True,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
    }
    
    val_loaders = {
        "real": DataLoader(
            val_datasets["real"],
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        ),
        "fake": DataLoader(
            val_datasets["fake"],
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=True
        )
    }
    
    ref_loader = DataLoader(
        ref_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = MMD_MIL_Detector(
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        dropout=config['dropout'],
        init_sigma=config['init_sigma']
    ).to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['lr'],
        weight_decay=config['weight_decay']
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config['num_epochs']
    )
    
    # Training loop
    print("\n" + "="*70)
    print("Starting Training")
    print("="*70)
    print(f"Train real: {len(train_datasets['real'])}")
    print(f"Train fake: {len(train_datasets['fake'])}")
    print(f"Val real: {len(val_datasets['real'])}")
    print(f"Val fake: {len(val_datasets['fake'])}")
    print(f"Reference: {len(ref_dataset)}")
    print("="*70)
    
    best_auroc = 0.0
    history = {
        'train_loss': [],
        'train_mmd': [],
        'val_auroc': [],
        'val_fpr95': []
    }
    
    for epoch in range(1, config['num_epochs'] + 1):
        # Train
        train_loss, train_mmd = train_epoch(
            model, train_loaders, optimizer, device, epoch
        )
        
        # Validate
        val_auroc, val_fpr95 = validate(
            model, val_loaders, ref_loader, device
        )
        
        # Step scheduler
        scheduler.step()
        
        # Log results
        print(f"\nEpoch {epoch}/{config['num_epochs']}:")
        print(f"  Train - Loss: {train_loss:.4f}, MMD²: {train_mmd:.6f}")
        print(f"  Val   - AUROC: {val_auroc:.4f}, FPR@95: {val_fpr95:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_mmd'].append(train_mmd)
        history['val_auroc'].append(val_auroc)
        history['val_fpr95'].append(val_fpr95)
        
        # Save best model
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auroc': val_auroc,
                'fpr95': val_fpr95,
                'config': config
            }, output_dir / 'best_model.pth')
            print(f"  ✓ Saved best model (AUROC: {val_auroc:.4f})")
        
        # Periodic checkpoint
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'auroc': val_auroc,
                'fpr95': val_fpr95,
                'config': config
            }, output_dir / f'checkpoint_epoch{epoch}.pth')
    
    # Save training history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=4)
    
    print("\n" + "="*70)
    print("Training Complete!")
    print("="*70)
    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Results saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()