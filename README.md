# TensorFlow Deep Learning Fundamentals

**Name:** Shashank Reddy D 

**ID:** 700781569



---

## Project Overview

This repository contains practical implementations of foundational Deep Learning operations using TensorFlow 2.x and Keras. The goal is to provide empirical insights into matrix manipulations, the behavior of multi-class vs. regression loss metrics, convergence differences between modern optimizers, and tracking experiment telemetry.

---

## Key Features & Tasks

### 1. Tensor Manipulations & Reshaping

* **Operations:** Demonstrates rank/shape extraction, tensor reshaping $(4, 6) \rightarrow (2, 3, 4)$, and axis transposing $(2, 3, 4) \rightarrow (3, 2, 4)$.
* **Broadcasting Mechanics:** Highlights how smaller tensors (e.g., shape `[1, 1, 4]`) are automatically stretched across missing dimensions during arithmetic operations to match target dimensions (`[3, 2, 4]`).

### 2. Loss Functions & Sensitivity Analysis

* **Implementation:** Manual and Keras-based calculation of **Mean Squared Error (MSE)** and **Categorical Cross-Entropy (CCE)** losses.
* **Comparative Behavior:** Analyzes sensitivity differences between quadratic regression loss (MSE) and logarithmic classification loss (CCE) under prediction degradation.

### 3. Optimizer Comparison on MNIST

* **Architectural Baseline:** Convolutional Neural Network (CNN) trained on normalized MNIST digit data.
* **Comparison:** Head-to-head convergence speed and validation accuracy comparison between **Adam** (adaptive learning rate) and **SGD** (Stochastic Gradient Descent).

### 4. Training & Telemetry via TensorBoard

* **Logging Setup:** Callbacks configured to stream training and validation loss/accuracy to `./logs/fit/`.
* **Diagnostics:** Analysis of training dynamics, identifying early indicators of model divergence and overfitting.

---

## Project Structure

```text
.
├── notebooks/
│   └── tensorflow_fundamentals.ipynb   # Main Jupyter Notebook
├── logs/
│   └── fit/                            # TensorBoard logs directory
├── requirements.txt                    # Project dependencies
└── README.md                           # Documentation

```

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the required packages via `pip`:

```bash
pip install tensorflow matplotlib numpy

```

---

## Usage Instructions

### Running the Notebook

Open and run the notebook in Jupyter or Google Colab:

```bash
jupyter notebook notebooks/tensorflow_fundamentals.ipynb

```

### Launching TensorBoard

To view real-time training metrics, scalar graphs, and model graphs, launch TensorBoard from your terminal:

```bash
tensorboard --logdir logs/fit

```

If you are running within a Jupyter Notebook cell:

```python
%load_ext tensorboard
%tensorboard --logdir logs/fit

```

---

## Results Summary

### Loss Function Sensitivity

When class prediction confidence drops on one-hot encoded targets:

* **Initial State:** `MSE = 0.0239` | `CCE = 0.2284`
* **Degraded Confidence:** `MSE = 0.0856` | `CCE = 0.4716`
* **Key Finding:** CCE heavily penalizes incorrect confident predictions exponentially, making it far better suited for classification than MSE.

### Optimizer Performance (5 Epochs)

| Optimizer | Final Validation Accuracy | Convergence Characteristics |
| --- | --- | --- |
| **Adam** | **98.47%** | Fast initial convergence; reaches high accuracy within 2-3 epochs. |
| **SGD** | **96.67%** | Steady, linear progression; requires more epochs or momentum tuning to match Adam. |

### TensorBoard Diagnostics

* **Overfitting Identification:** Divergence where `Training Loss` continues to fall while `Validation Loss` plateaus or increases.
* **Epoch Scaling:** Increasing epochs past the optimal validation loss threshold leads to memorization rather than generalization.
