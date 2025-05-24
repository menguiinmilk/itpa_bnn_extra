# ITER H-factor Predictor

A Bayesian Neural Network (BNN) based H-factor prediction tool using ITER parameters. Performs extrapolation through feature alignment techniques.

## Overview

This project provides a machine learning model that predicts H-factor under ITER operating conditions using the International Tokamak Physics Activity Global H-mode Database (ITPA Global H-mode Database). It quantifies prediction uncertainty through Bayesian neural network ensembles and offers an interactive GUI for real-time parameter adjustment and prediction visualization.

## Key Features

- **Bayesian Neural Network Ensemble**: Uncertainty quantification through 20-model ensemble
- **Feature Alignment**: Normalized feature alignment between existing tokamak data and ITER parameters
- **Interactive GUI**: PySide6-based real-time parameter adjustment and prediction interface
- **Uncertainty Estimation**: Separate calculation of epistemic and aleatoric uncertainties

## Input Parameters

### Variable Parameters
- **IP**: Plasma current (8-16 MA)
- **GWF**: Greenwald fraction (0-1.2)
- **DELTA**: Triangularity (0.2-0.8)
- **PNBI**: NBI heating power (0-100 MW)
- **PECRH**: ECRH heating power (0-100 MW)
- **PICRH**: ICRH heating power (0-100 MW)
- **PALPHA**: Alpha heating power (80 or 100 MW)
- **HIGHZ**: High-Z impurity presence
- **COAT**: Wall coating presence

### Fixed Parameters
- **BT**: Toroidal magnetic field (5.3 T)
- **RGEO**: Major radius (6.2 m)
- **AMIN**: Minor radius (2.0 m)
- **KAREA**: Elongation (1.7)
- **MEFF**: Effective mass (2.5)

## Usage

### GUI Execution
```bash
python itpa_bnn_gui.py
```

## Directory Structure

```
├── models/             # Neural network model definitions
├── utils/              # Utility functions
├── data/               # Data files
├── trained_models/     # Trained model storage
├── train_model.py      # Model training script
└── itpa_bnn_gui.py     # GUI application
```

## Dependencies

- JAX/Flax (neural network framework)
- PySide6 (GUI)
- scikit-learn (data preprocessing)
- pandas, numpy (data processing)

## Results

- **H-factor Prediction**: Predicted H-factor based on IPB98Y2 scaling
- **Uncertainty**: Epistemic and aleatoric uncertainties of predictions
- **Confidence Intervals**: 95% confidence interval provision 