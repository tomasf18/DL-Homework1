# Deep Learning Homework 1: Neural Network Fundamentals

### `Grade: 19.2/20`

This repository contains the implementation and analysis of fundamental neural network architectures for handwritten letter classification on the EMNIST dataset. Developed as Homework 1 for the Deep Learning course (25/26) at Instituto Superior Técnico, University of Lisbon.

## Overview

*Note: For detailed information, please refer to the [ProjectSpecification.pdf](ProjectSpecification.pdf).*

This project explores the progression from simple linear classifiers to deep neural networks, implementing and comparing:

**Question 1 - Classical Approaches (NumPy only, no ML libraries):**
- **Perceptron**: Basic linear classifier with online learning
- **Logistic Regression**: Multinomial logistic regression with L2 regularization and PCA feature engineering
- **Multi-Layer Perceptron (MLP)**: Fully implemented from scratch with manual backpropagation

**Question 2 - Modern Deep Learning (PyTorch):**
- **Feed-Forward Networks (FFN)**: Deep neural networks with systematic hyperparameter tuning and depth analysis

**Question 3 - Theoretical Demonstrations**

All models are trained and evaluated on the **EMNIST Letters dataset**, which contains grayscale images (28×28 pixels) of handwritten letters (A-Z, 26 classes).


## Key Results

*Note: For detailed analysis, methodology, and extended results, please refer to the [Report.pdf](Report.pdf).*

| Model | Implementation | Test Accuracy |
|-------|---------------|---------------|
| Perceptron | NumPy (from scratch) | 61.1% |
| Logistic Regression (Raw) | NumPy (from scratch) | 72.2% |
| Logistic Regression (PCA) | NumPy (from scratch) | 71.6% |
| Multi-Layer Perceptron | NumPy (from scratch) | 88.0% |
| FFN (Best 1-layer, width=128) | PyTorch | 89.5% |
| FFN (Best depth=3, width=32) | PyTorch | 86.5% |

## Project Structure

```
DL-Homework1/
├── README.md                           # This file
├── Report.pdf                          # Detailed analysis and results
├── prev_report_solutions/              # Reference materials
└── src/
    ├── data/
    │   └── emnist-letters.npz         # EMNIST dataset (28×28 grayscale images)
    │
    ├── utils.py                        # Utility functions (data loading, plotting, seeding)
    │
    ├── hw1-perceptron.py              # Q1.1: Perceptron implementation
    ├── hw1-logistic-regression.py     # Q1.2: Logistic Regression with PCA
    ├── hw1-multi-layer-perceptron.py  # Q1.3: MLP from scratch (NumPy)
    │
    ├── hw1_ffn_ex1.py                 # Q2.1: Basic FFN with PyTorch
    ├── hw1_ffn_ex2.py                 # Q2.2: Hyperparameter grid search
    ├── hw1_ffn_ex3.py                 # Q2.3: Network depth analysis
    │
    ├── q1-perceptron/                 # Perceptron results
    │   ├── Q1-perceptron-best.model
    │   └── Q1-perceptron-scores.json
    │
    ├── q1-logreg/                     # Logistic regression results
    │   ├── q1-logreg-best_*.model     # Best models for different configurations
    │   └── q1-logreg-scores.json      # Grid search results
    │
    ├── q1-mlp/                        # MLP results
    │   └── Q1-mlp-scores.json
    │
    ├── q2-ffn-2/                      # FFN hyperparameter search results
    │   ├── results_grid_search.csv    # Complete grid search results
    │   └── a_best_per_hidden_size.txt
    │
    └── q2-ffn-3/                      # FFN depth analysis results
        ├── results_depth_grid_search.csv
        ├── a_highest_val_acc_per_depth.txt
        └── b_best_depth_L3_test_accuracy.txt
```


## Usage

### Prerequisites

```bash
pip install numpy torch matplotlib
```

### Running Experiments

#### Perceptron
```bash
python src/hw1-perceptron.py \
    --epochs 20 \
    --data-path src/data/emnist-letters.npz \
    --seed 42
```

#### Logistic Regression
```bash
python src/hw1-logistic-regression.py \
    --epochs 20 \
    --data-path src/data/emnist-letters.npz \
    --seed 42
```

#### Multi-Layer Perceptron
```bash
python src/hw1-multi-layer-perceptron.py \
    --epochs 20 \
    --data-path src/data/emnist-letters.npz \
    --seed 42
```

#### Feed-Forward Network (Basic)
```bash
python src/hw1_ffn_ex1.py \
    --epochs 30 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --hidden-size 128 \
    --layers 3 \
    --activation relu \
    --dropout 0.0 \
    --data-path src/data/emnist-letters.npz
```

#### Hyperparameter Grid Search
```bash
python src/hw1_ffn_ex2.py --data-path src/data/emnist-letters.npz
```

#### Depth Analysis
```bash
python src/hw1_ffn_ex3.py --data-path src/data/emnist-letters.npz
```

## Authors

| <div align="center"><a href="https://github.com/tomasf18"><img src="https://avatars.githubusercontent.com/u/122024767?v=4" width="150px;" alt="Tomás Santos"/></a><br/><strong>Tomás Santos</strong><br/>116122<br/></div> | <div align="center"><a href="https://github.com/pedropmad"><img src="https://avatars.githubusercontent.com/u/163666619?v=4" width="150px;" alt="Pedro Duarte"/></a><br/><strong>Pedro Duarte</strong><br/>116390<br/></div> | <div align="center"><img src="https://avatars.githubusercontent.com/u/163666619?v=4" width="150px;" alt="Tiago Carvalho"/></a><br/><strong>Tiago Carvalho</strong><br/>106396<br/></div> |
| --- | --- | --- |

---
