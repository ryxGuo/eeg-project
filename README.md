# EEG Raw Data Classification Project

This repository contains code for classifying EEG raw signals using convolutional neural networks (CNNs) combined with density estimation techniques, such as Gaussian density modeling, RealNVP flows, and Density Softmax fusion.

## Folder Structure
- `grid_confs/` — Configuration files for running grid searches.
- `models.py` — Defines the models.
- `utils.py` — Utility functions for training and evaluation.
- `*.py` — Scripts for training, running grid search, Bayesian optimization, and evaluation.
- `.gitignore` — Specifies files and folders excluded from Git tracking.

## Requirements
- Python 3.10+
- PyTorch
- torcheval
- torchmetrics
- tqdm
- scikit-learn
- (other standard libraries like NumPy)

## Notes
- Results and model weights are **not included** to save space.
- Configuration files for grid search are located in `grid_confs/`.
- The `.gitignore` file ensures that large result files and datasets are excluded.