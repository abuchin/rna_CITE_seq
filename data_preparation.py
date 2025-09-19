#!/usr/bin/env python3
"""
Data Preparation for CrossModalFormer with Gene Expression Data

This script handles data loading, preprocessing, and preparation for gene expression analysis
using architectures similar to scvi-tools for single-cell RNA-seq data.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import anndata
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.sparse

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class for gene expression data preparation."""
    DATA_ROOT = "./data"
    EXPRESSION_FILE = "./data/expression_matrix.h5ad"  # AnnData file
    METADATA_FILE = "./data/metadata.csv"              # Cell metadata
    PROCESSED_DATA_DIR = "./data/processed"
    
    # Data processing parameters
    MIN_CELLS = 3                                       # Minimum cells expressing a gene
    MIN_GENES = 200                                     # Minimum genes per cell
    NORMALIZE = True                                    # Whether to normalize data
    LOG_TRANSFORM = True                                # Whether to log transform
    SCALE = True                                        # Whether to scale data
    
    # Model parameters
    LATENT_DIM = 10                                     # Latent dimension for VAE
    HIDDEN_DIM = 128                                    # Hidden layer dimension
    DROPOUT = 0.1                                       # Dropout rate
    
    # Training parameters
    BATCH_SIZE = 128                                    # Batch size
    NUM_WORKERS = 4                                     # Number of workers for data loading
    
    # Data split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15

def setup_directories():
    """Create necessary directories."""
    for dir_path in [Config.DATA_ROOT, Config.PROCESSED_DATA_DIR]:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    logger.info("Directories created successfully")

def load_gene_expression_data(expression_file: str, metadata_file: str = None) -> anndata.AnnData:
    """Load gene expression data from AnnData file or create sample data."""
    try:
        if os.path.exists(expression_file):
            # Load existing AnnData file
            adata = sc.read_h5ad(expression_file)
            logger.info(f"Loaded AnnData with shape: {adata.shape}")
            logger.info(f"Genes: {adata.n_vars}, Cells: {adata.n_obs}")
        else:
            logger.warning(f"Expression file not found: {expression_file}")
            adata = create_sample_expression_data()
            
        # Load metadata if provided
        if metadata_file and os.path.exists(metadata_file):
            metadata = pd.read_csv(metadata_file, index_col=0)
            # Ensure metadata matches cell indices
            common_cells = adata.obs.index.intersection(metadata.index)
            if len(common_cells) > 0:
                adata = adata[common_cells, :].copy()
                adata.obs = pd.concat([adata.obs, metadata.loc[common_cells]], axis=1)
                logger.info(f"Loaded metadata for {len(common_cells)} cells")
        
        return adata
        
    except Exception as e:
        logger.error(f"Error loading expression data: {e}")
        return create_sample_expression_data()

def create_sample_expression_data() -> anndata.AnnData:
    """Create sample gene expression data for testing."""
    logger.info("Creating sample gene expression data...")
    
    # Generate sample data
    n_cells = 1000
    n_genes = 2000
    
    # Create expression matrix (sparse-like with many zeros)
    expression_matrix = np.random.negative_binomial(5, 0.3, (n_cells, n_genes))
    # Add some zeros to simulate sparse data
    zero_mask = np.random.random((n_cells, n_genes)) < 0.7
    expression_matrix[zero_mask] = 0
    
    # Create gene names
    gene_names = [f"Gene_{i:04d}" for i in range(n_genes)]
    
    # Create cell metadata
    cell_metadata = pd.DataFrame({
        'cell_type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_cells),
        'condition': np.random.choice(['Control', 'Treatment'], n_cells),
        'batch': np.random.choice(['Batch_1', 'Batch_2'], n_cells),
        'n_genes': np.sum(expression_matrix > 0, axis=1),
        'total_counts': np.sum(expression_matrix, axis=1)
    }, index=[f"Cell_{i:04d}" for i in range(n_cells)])
    
    # Create AnnData object
    adata = anndata.AnnData(
        X=expression_matrix,
        var=pd.DataFrame(index=gene_names),
        obs=cell_metadata
    )
    
    logger.info(f"Created sample data with shape: {adata.shape}")
    return adata

def explore_gene_expression_data(adata: anndata.AnnData):
    """Explore the gene expression dataset structure and statistics."""
    logger.info("=== Gene Expression Dataset Overview ===")
    logger.info(f"Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    logger.info(f"Memory usage: {adata.n_obs * adata.n_vars * 8 / 1024**2:.2f} MB")
    
    logger.info("\n=== Expression Statistics ===")
    logger.info(f"Total counts: {np.sum(adata.X):,.0f}")
    logger.info(f"Mean counts per cell: {np.mean(np.sum(adata.X, axis=1)):.2f}")
    logger.info(f"Mean counts per gene: {np.mean(np.sum(adata.X, axis=0)):.2f}")
    
    # Sparsity
    sparsity = 1 - np.count_nonzero(adata.X) / adata.X.size
    logger.info(f"Data sparsity: {sparsity:.2%}")
    
    logger.info("\n=== Cell Metadata ===")
    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            unique_vals = adata.obs[col].nunique()
            logger.info(f"{col}: {unique_vals} unique values")
            if unique_vals <= 10:
                logger.info(f"  Values: {adata.obs[col].value_counts().to_dict()}")
        elif adata.obs[col].dtype in ['int64', 'float64']:
            logger.info(f"{col}: range {adata.obs[col].min():.2f} to {adata.obs[col].max():.2f}")
    
    # Quality control metrics
    logger.info("\n=== Quality Control Metrics ===")
    logger.info(f"Cells with < {Config.MIN_GENES} genes: {(adata.obs['n_genes'] < Config.MIN_GENES).sum()}")
    logger.info(f"Genes expressed in < {Config.MIN_CELLS} cells: {(np.sum(adata.X > 0, axis=0) < Config.MIN_CELLS).sum()}")

def preprocess_gene_expression(adata: anndata.AnnData) -> anndata.AnnData:
    """Preprocess gene expression data similar to scvi-tools."""
    logger.info("Starting gene expression preprocessing...")
    
    # Create a copy for processing
    adata_processed = adata.copy()
    
    # Quality control filtering
    logger.info("Applying quality control filters...")
    
    # Filter cells by minimum genes
    sc.pp.filter_cells(adata_processed, min_genes=Config.MIN_GENES)
    logger.info(f"After cell filtering: {adata_processed.shape[0]} cells")
    
    # Filter genes by minimum cells
    sc.pp.filter_genes(adata_processed, min_cells=Config.MIN_CELLS)
    logger.info(f"After gene filtering: {adata_processed.shape[1]} genes")
    
    # Normalize data
    if Config.NORMALIZE:
        logger.info("Normalizing data to counts per million...")
        sc.pp.normalize_total(adata_processed, target_sum=1e6)
    
    # Log transform
    if Config.LOG_TRANSFORM:
        logger.info("Applying log transformation...")
        sc.pp.log1p(adata_processed)
    
    # Scale data
    if Config.SCALE:
        logger.info("Scaling data...")
        sc.pp.scale(adata_processed, max_value=10)
    
    # Update metadata
    adata_processed.obs['n_genes'] = np.sum(adata_processed.X > 0, axis=1)
    adata_processed.obs['total_counts'] = np.sum(adata_processed.X, axis=1)
    
    logger.info("Gene expression preprocessing completed")
    return adata_processed

class GeneExpressionDataset(Dataset):
    """Custom dataset for gene expression data."""
    
    def __init__(self, adata: anndata.AnnData, batch_key: str = None, 
                 condition_key: str = None, apply_augmentation: bool = False):
        self.adata = adata
        self.batch_key = batch_key
        self.condition_key = condition_key
        self.apply_augmentation = apply_augmentation
        
        # Convert to dense if sparse
        if scipy.sparse.issparse(self.adata.X):
            self.adata.X = self.adata.X.toarray()
        
        # Get expression data
        self.expression_data = torch.FloatTensor(self.adata.X)
        
        # Get batch and condition labels if available
        self.batch_labels = None
        self.condition_labels = None
        
        if batch_key and batch_key in self.adata.obs.columns:
            self.batch_labels = torch.LongTensor(
                pd.Categorical(self.adata.obs[batch_key]).codes
            )
        
        if condition_key and condition_key in self.adata.obs.columns:
            self.condition_labels = torch.LongTensor(
                pd.Categorical(self.adata.obs[condition_key]).codes
            )
        
        logger.info(f"Dataset initialized with {len(self)} samples")
    
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        sample = {
            'expression': self.expression_data[idx],
            'index': idx
        }
        
        # Add batch labels if available
        if self.batch_labels is not None:
            sample['batch'] = self.batch_labels[idx]
        
        # Add condition labels if available
        if self.condition_labels is not None:
            sample['condition'] = self.condition_labels[idx]
        
        # Apply augmentation if requested
        if self.apply_augmentation:
            sample = self._apply_augmentation(sample)
        
        return sample
    
    def _apply_augmentation(self, sample: Dict) -> Dict:
        """Apply data augmentation techniques."""
        # Add small noise to expression values
        noise = torch.randn_like(sample['expression']) * 0.01
        sample['expression'] = sample['expression'] + noise
        
        # Random dropout of some genes (simulate missing data)
        dropout_mask = torch.rand_like(sample['expression']) > 0.05
        sample['expression'] = sample['expression'] * dropout_mask.float()
        
        return sample

def split_adata(adata: anndata.AnnData, train_ratio: float = 0.7, 
                val_ratio: float = 0.15, test_ratio: float = 0.15, 
                random_state: int = 42) -> Tuple[anndata.AnnData, anndata.AnnData, anndata.AnnData]:
    """Split AnnData into train, validation, and test sets."""
    
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Get indices
    n_cells = len(adata)
    indices = np.arange(n_cells)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    # Calculate split indices
    train_end = int(n_cells * train_ratio)
    val_end = train_end + int(n_cells * val_ratio)
    
    # Split data
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    train_adata = adata[train_indices, :].copy()
    val_adata = adata[val_indices, :].copy()
    test_adata = adata[test_indices, :].copy()
    
    logger.info(f"Dataset split: Train={len(train_adata)}, Val={len(val_adata)}, Test={len(test_adata)}")
    return train_adata, val_adata, test_adata

def create_data_loaders(train_adata: anndata.AnnData, val_adata: anndata.AnnData, 
                       test_adata: anndata.AnnData, batch_key: str = None, 
                       condition_key: str = None, batch_size: int = 128, 
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create data loaders for train, validation, and test sets."""
    
    # Create datasets
    train_dataset = GeneExpressionDataset(train_adata, batch_key, condition_key, apply_augmentation=True)
    val_dataset = GeneExpressionDataset(val_adata, batch_key, condition_key, apply_augmentation=False)
    test_dataset = GeneExpressionDataset(test_adata, batch_key, condition_key, apply_augmentation=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)
    
    logger.info(f"Data loaders created: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    return train_loader, val_loader, test_loader

def save_processed_data(train_adata: anndata.AnnData, val_adata: anndata.AnnData, 
                       test_adata: anndata.AnnData, save_dir: str):
    """Save processed data splits to disk."""
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save AnnData objects
    train_adata.write_h5ad(save_path / 'train.h5ad')
    val_adata.write_h5ad(save_path / 'val.h5ad')
    test_adata.write_h5ad(save_path / 'test.h5ad')
    
    # Save metadata
    metadata = {
        'train_cells': len(train_adata),
        'val_cells': len(val_adata),
        'test_cells': len(test_adata),
        'total_cells': len(train_adata) + len(val_adata) + len(test_adata),
        'genes': train_adata.n_vars,
        'obs_columns': list(train_adata.obs.columns),
        'var_columns': list(train_adata.var.columns),
        'config': {
            'min_cells': Config.MIN_CELLS,
            'min_genes': Config.MIN_GENES,
            'normalize': Config.NORMALIZE,
            'log_transform': Config.LOG_TRANSFORM,
            'scale': Config.SCALE,
            'latent_dim': Config.LATENT_DIM,
            'hidden_dim': Config.HIDDEN_DIM
        }
    }
    
    with open(save_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Processed data saved to: {save_path}")

def plot_data_statistics(adata: anndata.AnnData, save_path: str):
    """Plot data statistics and quality metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Gene counts distribution
    axes[0, 0].hist(adata.obs['n_genes'], bins=50, alpha=0.7)
    axes[0, 0].set_title('Genes per Cell')
    axes[0, 0].set_xlabel('Number of Genes')
    axes[0, 0].set_ylabel('Frequency')
    
    # Total counts distribution
    axes[0, 1].hist(adata.obs['total_counts'], bins=50, alpha=0.7)
    axes[0, 1].set_title('Total Counts per Cell')
    axes[0, 1].set_xlabel('Total Counts')
    axes[0, 1].set_ylabel('Frequency')
    
    # Expression sparsity
    sparsity_per_gene = 1 - np.count_nonzero(adata.X, axis=0) / adata.X.shape[0]
    axes[0, 2].hist(sparsity_per_gene, bins=50, alpha=0.7)
    axes[0, 2].set_title('Gene Sparsity')
    axes[0, 2].set_xlabel('Sparsity')
    axes[0, 2].set_ylabel('Frequency')
    
    # Cell type distribution (if available)
    if 'cell_type' in adata.obs.columns:
        cell_type_counts = adata.obs['cell_type'].value_counts()
        axes[1, 0].pie(cell_type_counts.values, labels=cell_type_counts.index, autopct='%1.1f%%')
        axes[1, 0].set_title('Cell Type Distribution')
    
    # Condition distribution (if available)
    if 'condition' in adata.obs.columns:
        condition_counts = adata.obs['condition'].value_counts()
        axes[1, 1].bar(condition_counts.index, condition_counts.values)
        axes[1, 1].set_title('Condition Distribution')
        axes[1, 1].set_ylabel('Count')
    
    # Batch distribution (if available)
    if 'batch' in adata.obs.columns:
        batch_counts = adata.obs['batch'].value_counts()
        axes[1, 2].bar(batch_counts.index, batch_counts.values)
        axes[1, 2].set_title('Batch Distribution')
        axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Data statistics plot saved to: {save_path}")

def main():
    """Main data preparation pipeline for gene expression data."""
    logger.info("Starting gene expression data preparation pipeline...")
    
    # Setup
    setup_directories()
    
    # Load data
    adata = load_gene_expression_data(Config.EXPRESSION_FILE, Config.METADATA_FILE)
    explore_gene_expression_data(adata)
    
    # Preprocess data
    adata_processed = preprocess_gene_expression(adata)
    
    # Plot statistics
    plot_data_statistics(adata_processed, os.path.join(Config.PROCESSED_DATA_DIR, 'data_statistics.png'))
    
    # Split data
    train_adata, val_adata, test_adata = split_adata(
        adata_processed, Config.TRAIN_RATIO, Config.VAL_RATIO, Config.TEST_RATIO
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_adata, val_adata, test_adata, 
        batch_key='batch', condition_key='condition',
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )
    
    # Save processed data
    save_processed_data(train_adata, val_adata, test_adata, Config.PROCESSED_DATA_DIR)
    
    # Test data loading
    logger.info("Testing data loading...")
    for batch in train_loader:
        logger.info(f"Batch shapes: expression={batch['expression'].shape}")
        if 'batch' in batch:
            logger.info(f"  batch labels: {batch['batch'].shape}")
        if 'condition' in batch:
            logger.info(f"  condition labels: {batch['condition'].shape}")
        break
    
    logger.info("Gene expression data preparation pipeline completed successfully!")
    logger.info("Data loaders are ready for model training.")

if __name__ == "__main__":
    main()
