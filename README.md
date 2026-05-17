# Part 1: Customer Churn Neural Network Analysis

## Overview
A feed-forward neural network to predict customer churn from a structured dataset (2000 rows, 17 columns). The dataset is severely imbalanced (98.45% retained, 1.55% churned), so **SMOTE** oversampling and **ROC-AUC / Churn Recall** are used as primary metrics instead of raw accuracy.

## Project Structure
```
part-1-neural-network-analysis/
├── README.md
├── notebook.ipynb
├── requirements.txt
└── results/
    ├── 01_target_distribution.png     # Task 1 — Feature exploration
    ├── 02_evaluation_outputs.png      # Task 4 — Confusion matrix, ROC, training curves
    ├── 03_all_experiments_curves.png  # Task 5 — All 6 experiment training curves
    ├── 04_model_comparison_table.png  # Task 5 — Bar chart comparison
    ├── 05_correlation_heatmap.png     # Correlation analysis
    └── model_comparison_table.csv    # Task 5 — Numeric results table
```

## Key Results

### Baseline Model
- Architecture: Dense(64, ReLU) → BN → Dropout(0.3) → Dense(32, ReLU) → BN → Dropout(0.3) → Dense(1, Sigmoid)
- Test Accuracy: **97.50%** | ROC-AUC: **0.8422** | Churn Recall: 16.7%

### Hyperparameter Comparison (Task 5)

| Experiment | Layers | LR | Batch | Activation | Test Acc | ROC-AUC | Churn Recall |
|---|---|---|---|---|---|---|---|
| Baseline | [64,32] | 0.001 | 32 | ReLU | 0.9750 | 0.8422 | 0.1667 |
| Shallow | [32] | 0.001 | 32 | ReLU | 0.9650 | 0.9205 | 0.1667 |
| **Deeper ✓** | **[128,64,32]** | **0.001** | **32** | **ReLU** | 0.7575 | 0.9201 | **1.0000** |
| High LR | [64,32] | 0.01 | 32 | ReLU | 0.9825 | 0.9399 | 0.1667 |
| Large Batch | [64,32] | 0.001 | 128 | ReLU | 0.7050 | 0.8003 | 0.6667 |
| **Tanh ✓** | **[64,32]** | **0.001** | **32** | **Tanh** | 0.4825 | **0.9543** | **1.0000** |

> 🏆 **Best for business use:** Deeper or Tanh models — both achieve 100% Churn Recall (catch every churning customer)

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Technologies
Python 3.10 | TensorFlow/Keras | scikit-learn | imbalanced-learn | pandas | matplotlib | seaborn
