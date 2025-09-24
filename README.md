# CrossModalFormer for CITE-Seq Data

A machine learning framework for predicting protein expression from single-cell RNA sequencing (scRNA-seq) data using CITE-Seq datasets. This project implements various regression models to establish relationships between gene expression and protein epitope measurements.

## Overview

This repository contains code for analyzing the relationship between measured gene expression (scRNA-seq) and protein epitopes (CITE-seq) using immunology datasets from the Allen Institute for Immunology. The framework includes data preprocessing, model training, and evaluation pipelines.

## Dataset

The project uses the **Allen Institute for Immunology** CITE-Seq dataset:
- **GEO Accession**: [GSE214546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE214546)
- **Data Type**: CITE-Seq (Cellular Indexing of Transcriptomes and Epitopes by Sequencing)
- **Species**: Human
- **Focus**: Immunology and immune cell analysis

## Project Structure

```
CrossModalFormer_CITE_Seq/
├── Code/
│   ├── CrossModalFormer/
│   │   ├── 01_data_preparation.ipynb    # Main analysis notebook
│   │   ├── data_preparation.py          # Data preprocessing pipeline
│   │   ├── model_training.py            # VAE model training
│   │   ├── requirements.txt             # Python dependencies
│   │   └── Data/
│   │       ├── documentation/           # Dataset documentation
│   │       │   └── Protein_expression.csv
│   │       ├── expression/              # RNA expression matrices
│   │       │   ├── RNA_expression_subset1a.csv
│   │       │   ├── RNA_expression_subset1a.h5ad
│   │       │   └── [other subsets]
│   │       ├── metadata/                # Cell metadata
│   │       │   └── RNA_metadata.csv
│   │       └── cluster/                 # Clustering results
│   │           └── RNA_UMAP_cluster.csv
│   └── code_basics/                     # Setup utilities
└── README.md                           # This file
```

## Features

### Data Processing
- **RNA-seq Preprocessing**: Normalization, log transformation, highly variable gene selection
- **Protein Data Processing**: Standard scaling and normalization
- **Cell Matching**: Identification of common cells between RNA and protein datasets
- **Quality Control**: Filtering and validation of data quality

### Models Implemented

1. **Ridge Regression** (R² = 0.249)
   - Linear regression with L2 regularization
   - Baseline model for comparison

2. **Partial Least Squares (PLS)** (R² = 0.148)
   - Dimensionality reduction with supervised learning
   - 10 components used

3. **XGBoost** (R² = 0.458)
   - Gradient boosting with multi-output regression
   - Best performing model
   - Individual protein predictions available

### Protein Targets

The model predicts expression for 20 protein epitopes:
- CD117, CD119, CD140a, CD140b, CD172a, CD184, CD202b, CD274
- CD29, CD309, CD44, CD47, CD49f, CD58, CD59, CD61
- HLA_A, HLA_E, CD9, CD279

Best predicted proteins (R² > 0.5):
- CD49f (R² = 0.644)
- CD29 (R² = 0.613)
- CD44 (R² = 0.589)
- CD274 (R² = 0.599)

## Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for VAE training)

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd CrossModalFormer_CITE_Seq
```

2. **Install dependencies**:
```bash
cd Code/CrossModalFormer
pip install -r requirements.txt
```

3. **Verify installation**:
```bash
python -c "import scanpy, torch, xgboost; print('All dependencies installed successfully')"
```

## Usage

### Quick Start

1. **Run the main analysis notebook**:
```bash
jupyter notebook 01_data_preparation.ipynb
```

2. **Execute data preprocessing pipeline**:
```bash
python data_preparation.py
```

3. **Train VAE models**:
```bash
python model_training.py
```

### Data Preparation

The notebook `01_data_preparation.ipynb` contains the complete analysis pipeline:

1. **Data Import**: Loads RNA-seq and protein expression data
2. **Preprocessing**: Normalizes and filters data
3. **Cell Matching**: Finds common cells between modalities
4. **Model Training**: Trains and evaluates multiple regression models
5. **Visualization**: Creates prediction plots and performance metrics

### Configuration

Key parameters can be modified in the configuration classes:

- **Data Processing**: `MIN_CELLS`, `MIN_GENES`, normalization settings
- **Model Parameters**: `LATENT_DIM`, `HIDDEN_DIM`, `DROPOUT`
- **Training**: `EPOCHS`, `LEARNING_RATE`, `BATCH_SIZE`

## Results

### Model Performance

| Model | Average R² | Best Protein | Worst Protein |
|-------|------------|--------------|---------------|
| Ridge Regression | 0.249 | CD49f (0.644) | CD279 (0.359) |
| Partial Least Squares | 0.148 | CD49f (0.644) | CD279 (0.359) |
| XGBoost | 0.458 | CD49f (0.644) | CD279 (0.359) |

### Key Findings

- **XGBoost** provides the best overall performance with 45.8% explained variance
- **CD49f** is the most predictable protein across all models
- **CD279** (PD-1) shows the lowest predictability
- Linear models (Ridge, PLS) show similar performance patterns

## Technical Details

### Data Format
- **RNA Expression**: Sparse matrices in H5AD format (AnnData)
- **Protein Expression**: CSV files with cell IDs as columns
- **Metadata**: Cell annotations and experimental conditions

### Preprocessing Steps
1. Total count normalization (10,000 counts per cell)
2. Log(x+1) transformation
3. Highly variable gene selection (top 3,000 genes)
4. Standard scaling for protein data

### Model Architecture
- **VAE Framework**: Encoder-decoder architecture with latent representation
- **Batch Correction**: Handles technical batch effects
- **Multi-modal Learning**: Joint representation of RNA and protein data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this code in your research, please cite:

```bibtex
@software{crossmodalformer_cite_seq,
  title={CrossModalFormer for CITE-Seq Data},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-username]/CrossModalFormer_CITE_Seq}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Allen Institute for Immunology** for providing the CITE-Seq dataset
- **scvi-tools** community for inspiration on single-cell analysis frameworks
- **XGBoost** developers for the gradient boosting implementation

## Contact

For questions or support, please open an issue on GitHub or contact [anat.buchin@gmail.com].

---

**Note**: This project is for research purposes. Please ensure you have appropriate permissions to use the Allen Institute dataset and follow their data usage guidelines.
