#!/usr/bin/env python3
"""
Model Training for CrossModalFormer with Gene Expression Data

This script implements the training pipeline for VAE models similar to scvi-tools
for single-cell RNA-seq data analysis.
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import time
from datetime import datetime
import scanpy as sc
import anndata
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingConfig:
    """Configuration class for gene expression model training."""
    
    # Model parameters
    INPUT_DIM = None  # Will be set from data
    HIDDEN_DIM = 128
    LATENT_DIM = 10
    NUM_BATCHES = None  # Will be set from data
    NUM_CONDITIONS = None  # Will be set from data
    DROPOUT = 0.1
    
    # VAE parameters
    KL_WEIGHT = 1.0
    RECONSTRUCTION_LOSS = 'negative_binomial'  # or 'gaussian'
    
    # Training parameters
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    BATCH_SIZE = 128
    GRADIENT_CLIP = 1.0
    
    # Paths
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    DATA_DIR = "./data/processed"
    
    # Early stopping
    PATIENCE = 15
    MIN_DELTA = 0.001
    
    # Data loading
    NUM_WORKERS = 4
    PIN_MEMORY = True

class Encoder(nn.Module):
    """Encoder network for VAE."""
    
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, 
                 num_batches: int, num_conditions: int, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_batches = num_batches
        self.num_conditions = num_conditions
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Batch and condition embeddings
        if num_batches > 1:
            self.batch_embedding = nn.Embedding(num_batches, hidden_dim)
        if num_conditions > 1:
            self.condition_embedding = nn.Embedding(num_conditions, hidden_dim)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)
        
        # Additional outputs for batch correction
        self.fc_batch = nn.Linear(hidden_dim, num_batches) if num_batches > 1 else None
        self.fc_condition = nn.Linear(hidden_dim, num_conditions) if num_conditions > 1 else None
        
    def forward(self, x: torch.Tensor, batch_labels: Optional[torch.Tensor] = None, 
                condition_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through encoder."""
        
        # Encode input
        encoded = self.encoder(x)
        
        # Add batch and condition information
        if batch_labels is not None and self.num_batches > 1:
            batch_emb = self.batch_embedding(batch_labels)
            encoded = encoded + batch_emb
        
        if condition_labels is not None and self.num_conditions > 1:
            condition_emb = self.condition_embedding(condition_labels)
            encoded = encoded + condition_emb
        
        # Latent space
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        
        # Sample from latent space
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # Additional outputs
        outputs = {
            'z': z,
            'mu': mu,
            'log_var': log_var,
            'encoded': encoded
        }
        
        if self.fc_batch is not None:
            outputs['batch_logits'] = self.fc_batch(encoded)
        
        if self.fc_condition is not None:
            outputs['condition_logits'] = self.fc_condition(encoded)
        
        return outputs

class Decoder(nn.Module):
    """Decoder network for VAE."""
    
    def __init__(self, latent_dim: int, hidden_dim: int, output_dim: int, 
                 num_batches: int, num_conditions: int, dropout: float = 0.1):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_batches = num_batches
        self.num_conditions = num_conditions
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Batch and condition embeddings for decoder
        if num_batches > 1:
            self.batch_embedding = nn.Embedding(num_batches, output_dim)
        if num_conditions > 1:
            self.condition_embedding = nn.Embedding(num_conditions, output_dim)
        
        # Output parameters for negative binomial
        if TrainingConfig.RECONSTRUCTION_LOSS == 'negative_binomial':
            self.theta = nn.Parameter(torch.ones(output_dim))
            self.ridge_lambda = 1e-5
    
    def forward(self, z: torch.Tensor, batch_labels: Optional[torch.Tensor] = None, 
                condition_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through decoder."""
        
        # Decode latent representation
        decoded = self.decoder(z)
        
        # Add batch and condition information
        if batch_labels is not None and self.num_batches > 1:
            batch_emb = self.batch_embedding(batch_labels)
            decoded = decoded + batch_emb
        
        if condition_labels is not None and self.num_conditions > 1:
            condition_emb = self.condition_embedding(condition_labels)
            decoded = decoded + condition_emb
        
        outputs = {'decoded': decoded}
        
        if TrainingConfig.RECONSTRUCTION_LOSS == 'negative_binomial':
            # For negative binomial, we need rate and dispersion
            rate = torch.exp(decoded)
            theta = torch.clamp(self.theta, min=1e-6)
            outputs.update({
                'rate': rate,
                'theta': theta
            })
        else:
            # For Gaussian, decoded is the mean
            outputs['mean'] = decoded
        
        return outputs

class GeneExpressionVAE(nn.Module):
    """Variational Autoencoder for gene expression data, similar to scvi-tools."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, latent_dim: int = 10, 
                 num_batches: int = 1, num_conditions: int = 1, dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_batches = num_batches
        self.num_conditions = num_conditions
        
        # Encoder and decoder
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, num_batches, num_conditions, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, num_batches, num_conditions, dropout)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"GeneExpressionVAE initialized with {sum(p.numel() for p in self.parameters()):,} parameters")
        logger.info(f"Input dim: {input_dim}, Hidden dim: {hidden_dim}, Latent dim: {latent_dim}")
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, batch_labels: Optional[torch.Tensor] = None, 
                condition_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the VAE."""
        
        # Encode
        encoder_outputs = self.encoder(x, batch_labels, condition_labels)
        
        # Decode
        decoder_outputs = self.decoder(encoder_outputs['z'], batch_labels, condition_labels)
        
        # Combine outputs
        outputs = {**encoder_outputs, **decoder_outputs}
        
        return outputs
    
    def encode(self, x: torch.Tensor, batch_labels: Optional[torch.Tensor] = None, 
               condition_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode data to latent space."""
        with torch.no_grad():
            encoder_outputs = self.encoder(x, batch_labels, condition_labels)
            return encoder_outputs['z']
    
    def decode(self, z: torch.Tensor, batch_labels: Optional[torch.Tensor] = None, 
               condition_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode from latent space."""
        with torch.no_grad():
            decoder_outputs = self.decoder(z, batch_labels, condition_labels)
            return decoder_outputs['decoded']

class VAELoss(nn.Module):
    """Loss function for VAE training."""
    
    def __init__(self, kl_weight: float = 1.0, reconstruction_loss: str = 'negative_binomial'):
        super().__init__()
        self.kl_weight = kl_weight
        self.reconstruction_loss = reconstruction_loss
        
        # Classification losses
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor, 
                batch_labels: Optional[torch.Tensor] = None, 
                condition_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute VAE loss."""
        
        # KL divergence loss
        mu = outputs['mu']
        log_var = outputs['log_var']
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Reconstruction loss
        if self.reconstruction_loss == 'negative_binomial':
            recon_loss = self._negative_binomial_loss(outputs, targets)
        else:
            recon_loss = self._gaussian_loss(outputs, targets)
        
        # Classification losses
        classification_losses = {}
        if 'batch_logits' in outputs and batch_labels is not None:
            classification_losses['batch'] = self.ce_loss(outputs['batch_logits'], batch_labels)
        
        if 'condition_logits' in outputs and condition_labels is not None:
            classification_losses['condition'] = self.ce_loss(outputs['condition_logits'], condition_labels)
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        for loss_name, loss_val in classification_losses.items():
            total_loss = total_loss + loss_val
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss,
            'classification_losses': classification_losses
        }
    
    def _negative_binomial_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Negative binomial reconstruction loss."""
        rate = outputs['rate']
        theta = outputs['theta']
        
        # Clamp values for numerical stability
        rate = torch.clamp(rate, min=1e-8)
        theta = torch.clamp(theta, min=1e-8)
        
        # Negative binomial log-likelihood
        log_theta_mu = torch.log(theta + rate)
        log_theta_mu_1 = torch.log(theta + 1)
        
        log_ll = (torch.lgamma(theta + targets) - torch.lgamma(theta) - torch.lgamma(targets + 1) +
                 theta * log_theta_mu + targets * torch.log(rate) - (theta + targets) * log_theta_mu)
        
        return -torch.mean(log_ll)
    
    def _gaussian_loss(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        """Gaussian reconstruction loss."""
        mean = outputs['mean']
        return torch.mean(0.5 * (targets - mean).pow(2))

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """Check if early stopping should be triggered."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

class TrainingMetrics:
    """Track and store training metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.learning_rates = []
        self.epoch_times = []
    
    def update(self, train_loss: float, val_loss: float, recon_loss: float, 
               kl_loss: float, learning_rate: float, epoch_time: float):
        """Update metrics for current epoch."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.recon_losses.append(recon_loss)
        self.kl_losses.append(kl_loss)
        self.learning_rates.append(learning_rate)
        self.epoch_times.append(epoch_time)
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best validation metrics."""
        best_epoch = np.argmin(self.val_losses)
        return {
            'best_epoch': best_epoch + 1,
            'best_val_loss': self.val_losses[best_epoch],
            'corresponding_train_loss': self.train_losses[best_epoch]
        }
    
    def plot_training_curves(self, save_path: str):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Total loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Validation Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Reconstruction loss
        axes[0, 1].plot(self.recon_losses, color='green')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
        
        # KL divergence loss
        axes[0, 2].plot(self.kl_losses, color='orange')
        axes[0, 2].set_title('KL Divergence Loss')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Loss')
        axes[0, 2].grid(True)
        
        # Learning rate
        axes[1, 0].plot(self.learning_rates, color='purple')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Epoch times
        axes[1, 1].plot(self.epoch_times, color='brown')
        axes[1, 1].set_title('Epoch Training Time')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].grid(True)
        
        # Loss components
        axes[1, 2].plot(self.recon_losses, label='Reconstruction', color='green')
        axes[1, 2].plot(self.kl_losses, label='KL Divergence', color='orange')
        axes[1, 2].set_title('Loss Components')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Training curves saved to: {save_path}")

def train_epoch(model: nn.Module, train_loader: DataLoader, criterion: nn.Module, 
                optimizer: optim.Optimizer, device: torch.device) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        # Move data to device
        expression = batch['expression'].to(device)
        batch_labels = batch.get('batch', None)
        condition_labels = batch.get('condition', None)
        
        if batch_labels is not None:
            batch_labels = batch_labels.to(device)
        if condition_labels is not None:
            condition_labels = condition_labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(expression, batch_labels, condition_labels)
        
        # Compute loss
        loss_dict = criterion(outputs, expression, batch_labels, condition_labels)
        total_loss += loss_dict['total_loss'].item()
        total_recon_loss += loss_dict['reconstruction_loss'].item()
        total_kl_loss += loss_dict['kl_loss'].item()
        
        # Backward pass
        loss_dict['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), TrainingConfig.GRADIENT_CLIP)
        optimizer.step()
        
        # Update progress bar
        pbar.set_postfix({
            'Total Loss': f"{loss_dict['total_loss'].item():.4f}",
            'Recon Loss': f"{loss_dict['reconstruction_loss'].item():.4f}",
            'KL Loss': f"{loss_dict['kl_loss'].item():.4f}"
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    avg_kl_loss = total_kl_loss / len(train_loader)
    
    return {
        'total_loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss
    }

def validate_epoch(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, 
                  device: torch.device) -> Dict[str, float]:
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            expression = batch['expression'].to(device)
            batch_labels = batch.get('batch', None)
            condition_labels = batch.get('condition', None)
            
            if batch_labels is not None:
                batch_labels = batch_labels.to(device)
            if condition_labels is not None:
                condition_labels = condition_labels.to(device)
            
            # Forward pass
            outputs = model(expression, batch_labels, condition_labels)
            
            # Compute loss
            loss_dict = criterion(outputs, expression, batch_labels, condition_labels)
            total_loss += loss_dict['total_loss'].item()
            total_recon_loss += loss_dict['reconstruction_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
    
    avg_loss = total_loss / len(val_loader)
    avg_recon_loss = total_recon_loss / len(val_loader)
    avg_kl_loss = total_kl_loss / len(val_loader)
    
    return {
        'total_loss': avg_loss,
        'reconstruction_loss': avg_recon_loss,
        'kl_loss': avg_kl_loss
    }

def load_data_loaders(data_dir: str, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Load data loaders from processed data directory."""
    try:
        # Try to import from data preparation
        import sys
        sys.path.append('.')
        from data_preparation import GeneExpressionDataset
        
        # Load processed data
        train_adata = sc.read_h5ad(os.path.join(data_dir, 'train.h5ad'))
        val_adata = sc.read_h5ad(os.path.join(data_dir, 'val.h5ad'))
        test_adata = sc.read_h5ad(os.path.join(data_dir, 'test.h5ad'))
        
        # Create datasets
        train_dataset = GeneExpressionDataset(train_adata, batch_key='batch', condition_key='condition', apply_augmentation=True)
        val_dataset = GeneExpressionDataset(val_adata, batch_key='batch', condition_key='condition', apply_augmentation=False)
        test_dataset = GeneExpressionDataset(test_adata, batch_key='batch', condition_key='condition', apply_augmentation=False)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                num_workers=num_workers, pin_memory=TrainingConfig.PIN_MEMORY)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                               num_workers=num_workers, pin_memory=TrainingConfig.PIN_MEMORY)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                                num_workers=num_workers, pin_memory=TrainingConfig.PIN_MEMORY)
        
        # Set configuration from data
        TrainingConfig.INPUT_DIM = train_adata.n_vars
        TrainingConfig.NUM_BATCHES = len(train_adata.obs['batch'].unique()) if 'batch' in train_adata.obs.columns else 1
        TrainingConfig.NUM_CONDITIONS = len(train_adata.obs['condition'].unique()) if 'condition' in train_adata.obs.columns else 1
        
        logger.info(f"Data loaders loaded: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
        logger.info(f"Input dim: {TrainingConfig.INPUT_DIM}, Batches: {TrainingConfig.NUM_BATCHES}, Conditions: {TrainingConfig.NUM_CONDITIONS}")
        return train_loader, val_loader, test_loader
        
    except ImportError:
        logger.warning("Could not import from data_preparation. Creating dummy data loaders.")
        return create_dummy_data_loaders(batch_size, num_workers)

def create_dummy_data_loaders(batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create dummy data loaders for testing purposes."""
    class DummyDataset:
        def __init__(self, size: int = 100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'expression': torch.randn(1000),  # 1000 genes
                'batch': torch.randint(0, 2, (1,)).squeeze(),
                'condition': torch.randint(0, 2, (1,)).squeeze()
            }
    
    train_dataset = DummyDataset(1000)
    val_dataset = DummyDataset(200)
    test_dataset = DummyDataset(200)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Set dummy configuration
    TrainingConfig.INPUT_DIM = 1000
    TrainingConfig.NUM_BATCHES = 2
    TrainingConfig.NUM_CONDITIONS = 2
    
    logger.info("Dummy data loaders created for testing")
    return train_loader, val_loader, test_loader

def save_checkpoint(model: nn.Module, optimizer: optim.Optimizer, epoch: int, 
                   val_loss: float, save_path: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': TrainingConfig.__dict__
    }
    
    torch.save(checkpoint, save_path)
    logger.info(f"Checkpoint saved to: {save_path}")

def main():
    """Main training pipeline for gene expression VAE."""
    logger.info("Starting Gene Expression VAE training pipeline...")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create directories
    for dir_path in [TrainingConfig.CHECKPOINT_DIR, TrainingConfig.LOG_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader, test_loader = load_data_loaders(
        TrainingConfig.DATA_DIR, 
        TrainingConfig.BATCH_SIZE, 
        TrainingConfig.NUM_WORKERS
    )
    
    # Initialize model
    model = GeneExpressionVAE(
        input_dim=TrainingConfig.INPUT_DIM,
        hidden_dim=TrainingConfig.HIDDEN_DIM,
        latent_dim=TrainingConfig.LATENT_DIM,
        num_batches=TrainingConfig.NUM_BATCHES,
        num_conditions=TrainingConfig.NUM_CONDITIONS,
        dropout=TrainingConfig.DROPOUT
    ).to(device)
    
    # Loss and optimizer
    criterion = VAELoss(
        kl_weight=TrainingConfig.KL_WEIGHT,
        reconstruction_loss=TrainingConfig.RECONSTRUCTION_LOSS
    )
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TrainingConfig.LEARNING_RATE,
        weight_decay=TrainingConfig.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=TrainingConfig.PATIENCE, 
        min_delta=TrainingConfig.MIN_DELTA
    )
    
    # Training metrics
    metrics = TrainingMetrics()
    
    # Training loop
    logger.info(f"Starting training for {TrainingConfig.EPOCHS} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(TrainingConfig.EPOCHS):
        epoch_start_time = time.time()
        
        logger.info(f"\nEpoch {epoch+1}/{TrainingConfig.EPOCHS}")
        logger.info("-" * 50)
        
        # Training
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation
        val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_metrics['total_loss'])
        
        # Epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Update metrics
        metrics.update(
            train_metrics['total_loss'], 
            val_metrics['total_loss'],
            train_metrics['reconstruction_loss'],
            train_metrics['kl_loss'],
            optimizer.param_groups[0]['lr'], 
            epoch_time
        )
        
        # Log results
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f}")
        logger.info(f"Val Loss: {val_metrics['total_loss']:.4f}")
        logger.info(f"Recon Loss: {train_metrics['reconstruction_loss']:.4f}")
        logger.info(f"KL Loss: {train_metrics['kl_loss']:.4f}")
        logger.info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        logger.info(f"Epoch Time: {epoch_time:.2f}s")
        
        # Save best model
        if val_metrics['total_loss'] < best_val_loss:
            best_val_loss = val_metrics['total_loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics['total_loss'],
                os.path.join(TrainingConfig.CHECKPOINT_DIR, 'best_model.pth')
            )
            logger.info(f"New best model saved with validation loss: {val_metrics['total_loss']:.4f}")
        
        # Early stopping check
        if early_stopping(val_metrics['total_loss']):
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Training completed
    logger.info("\nTraining completed!")
    
    # Plot training curves
    metrics.plot_training_curves(os.path.join(TrainingConfig.LOG_DIR, 'training_curves.png'))
    
    # Load best model and evaluate on test set
    logger.info("Loading best model for final evaluation...")
    checkpoint = torch.load(os.path.join(TrainingConfig.CHECKPOINT_DIR, 'best_model.pth'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_metrics = validate_epoch(model, test_loader, criterion, device)
    logger.info(f"\nFinal Test Results:")
    logger.info(f"Test Loss: {test_metrics['total_loss']:.4f}")
    logger.info(f"Test Reconstruction Loss: {test_metrics['reconstruction_loss']:.4f}")
    logger.info(f"Test KL Loss: {test_metrics['kl_loss']:.4f}")
    
    # Save final results
    best_metrics = metrics.get_best_metrics()
    results = {
        'best_epoch': best_metrics['best_epoch'],
        'best_val_loss': best_metrics['best_val_loss'],
        'test_loss': test_metrics['total_loss'],
        'test_reconstruction_loss': test_metrics['reconstruction_loss'],
        'test_kl_loss': test_metrics['kl_loss'],
        'training_epochs': len(metrics.train_losses),
        'final_train_loss': metrics.train_losses[-1],
        'final_val_loss': metrics.val_losses[-1],
        'model_config': {
            'input_dim': TrainingConfig.INPUT_DIM,
            'hidden_dim': TrainingConfig.HIDDEN_DIM,
            'latent_dim': TrainingConfig.LATENT_DIM,
            'num_batches': TrainingConfig.NUM_BATCHES,
            'num_conditions': TrainingConfig.NUM_CONDITIONS,
            'reconstruction_loss': TrainingConfig.RECONSTRUCTION_LOSS
        },
        'training_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(TrainingConfig.LOG_DIR, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nResults saved to {TrainingConfig.LOG_DIR}/training_results.json")
    logger.info(f"Best model saved to {TrainingConfig.CHECKPOINT_DIR}/best_model.pth")
    logger.info("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
