# Neural Network Forward Modeling of Mechanical MNIST: Strain Energy and Displacement Field Prediction

**Author:** Andy Kapoor

## Repository Structure
```
.
├── Mech_MNIST_ForwardModeling.ipynb  # Main analysis notebook
├── data/
│   ├── mnist_img_train.txt.zip       # MNIST training bitmap data
│   └── mnist_img_test.txt.zip        # MNIST test bitmap data
└── README.md
```

> **Note:** The `FEA_displacement_results_step12/` folder is not included in this repository due to size. Download it from the Mechanical MNIST GitHub: https://github.com/elejeune11/Mechanical-MNIST and place it in the root directory before running the notebook.

---

## Overview

This project applies multilayer perceptron (MLP) neural networks to two forward modeling problems on the Mechanical MNIST dataset. In Problem 1, a bitmap image is used to predict the scalar change in strain energy at 50% uniaxial extension (step 12). In Problem 2, the same bitmap input is used to predict the full-field x and y displacement outputs at step 12, and the predicted displacement fields are converted to strain fields via finite difference approximation. Together these two problems explore how well neural network metamodels can learn the mapping from material geometry to mechanical response.

---

## Dataset

**Mechanical MNIST** is a benchmark dataset introduced by Lejeune (2020) consisting of 70,000 finite element simulations of heterogeneous material domains under large deformation. Each sample pairs an MNIST bitmap image (used to define material properties on a 28x28 grid) with the resulting FEA outputs including strain energy and full-field displacements.

- Paper: Lejeune, E. (2020). Mechanical MNIST: A benchmark dataset for mechanical metamodels. *Extreme Mechanics Letters*. https://doi.org/10.1016/j.eml.2020.100659
- Dataset: https://open.bu.edu/handle/2144/39371
- GitHub: https://github.com/elejeune11/Mechanical-MNIST

---

## Problem

Given a 28x28 bitmap image encoding the material property distribution of a heterogeneous domain, predict the mechanical response produced by FEA simulation under uniaxial extension to 50% deformation. Two targets are explored: a single scalar (strain energy) and a full spatial field (x and y displacements).

---

## Data

| Dataset | Description |
|---|---|
| MNIST bitmaps | 60,000 train / 10,000 test 28x28 images encoding material properties |
| Strain energy (psi) | Scalar change in strain energy at step 12 (50% extension), target for Problem 1 |
| FEA displacement fields | x and y displacement outputs at 784 nodes per image (1,568 features total) at step 12, target for Problem 2 |

---

## Methods

### Problem 1: Strain Energy Prediction
- **MLP:** 3 hidden layers (128 --> 64 --> 32 neurons)
- Input: 784-feature flattened bitmap, StandardScaler normalized
- Output: scalar strain energy at step 12
- Activation: ReLU. Optimizer: Adam. Early stopping with 10% validation split.
- Converged after 55 epochs (200 max). R² = 0.823 on test set.

### Problem 2: Full-Field Displacement Prediction
- **MLP:** 3 hidden layers (256 --> 128 --> 64 neurons)
- Input: 784-feature flattened bitmap, StandardScaler normalized
- Output: 1,568-feature displacement field (ux and uy concatenated)
- Activation: ReLU. Optimizer: Adam. Early stopping with 10% validation split.
- Converged after 81 epochs. Test MAE = 0.101412 bitmap length units.
- Predicted displacement fields converted to strain fields (exx, eyy, exy) using np.gradient finite difference approximation.

---

## Results Summary

| Problem | Metric | Value |
|---|---|---|
| Strain energy prediction | R² (test set) | 0.823 |
| Displacement field prediction | MAE (test set) | 0.101412 bitmap length units |

Global displacement field structure is recovered well. Local strain gradients show slightly more error, particularly in exx, as spatial differentiation amplifies small local prediction errors.

---

## Dependencies
```
numpy
matplotlib
scipy
scikit-learn
torch
tqdm
requests
```

Install with:
```bash
pip install numpy matplotlib scipy scikit-learn torch tqdm requests
```

---

## Limitations & Future Work

- Larger hidden layers improved accuracy at the cost of training time on CPU. GPU training would enable larger architectures.
- Strain field predictions amplify local displacement errors. Direct strain prediction as a target may reduce this effect.
- Future work: convolutional architectures to better exploit spatial structure of bitmap inputs, multi-step prediction across all deformation steps, uncertainty quantification.
