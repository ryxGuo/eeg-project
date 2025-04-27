# Density-Aware Modeling for EEG-Based MCI Detection

This repository contains code for detecting mild cognitive impairment (MCI) from EEG signals by leveraging uncertainty-aware density estimation techniques.
The goal is to enhance classification robustness, interpretability, and trustworthiness for clinical applications.

## Project Overview
- Use a CNN backbone to extract latent features from time-domain EEG segments.
- Apply density estimation methods to model the latent feature space:
  - **Gaussian density estimation** (simple likelihood modeling)
  - **RealNVP normalizing flow** (flexible density modeling with coupling layers, masking strategies, and prior distributions)
  - **Density-Softmax fusion** (likelihood-aware calibration of softmax predictions)
- Explore uncertainty evaluation and trust calibration through likelihood thresholding.

## Dataset
- **GENEEG**: Real-world EEG dataset collected in clinical settings
  - 28 MCI patients and 35 cognitively normal controls
  - 4-minute resting-state EEG, 19 channels (2 channels removed)
- **Preprocessing**:
  - Band-pass filtering (5–20 Hz)
  - Segmentation into 1-second windows (200 time steps)
  - Patient-level split: 75% train, 25% test

## Code Structure
- `grid_confs/` — Hyperparameter configuration files for grid search experiments
- `grid_search/` — Scripts for hyperparameter tuning
- `models.py` — CNN feature extractor, RealNVP flow model
- `utils.py` — Helper functions for training, evaluation, calibration

## Key Techniques
- **CNN Feature Extraction**: 3 convolutional layers + 2 fully connected layers trained with cross-entropy loss
- **RealNVP Flow**: Alternating binary masking, random block masking; Gaussian, Laplace, and Gaussian mixture priors
- **Density-Softmax Fusion**: Likelihoods used to rescale CNN logits at test-time for calibrated predictions
- **Uncertainty Evaluation**: Identify unreliable predictions by thresholding likelihood estimates

## Requirements
- Python 3.10+
- PyTorch
- torcheval
- torchmetrics
- tqdm
- scikit-learn
- (other standard libraries like NumPy)

## Notes
- Intermediate result folders (weights, large result CSVs) are excluded by `.gitignore`.
- Only the necessary code and configuration files are included for reproducibility.