# JAX-Based MLIP Framework

This repository implements **a JAX-based framework for data-efficient machine learned interatomic potentials (MLIPs)**.  
The project focuses on developing a lightweight neural-network potential that can learn atomic interactions with high data efficiency while maintaining differentiability and computational performance through the JAX ecosystem.

The framework is designed to explore how physically motivated features and compact neural architectures can improve the learning efficiency of MLIPs. In particular, we investigate the effect of incorporating additional geometric descriptors (such as dihedral angles) into the neural network representation of atomic environments.

---

## Repository Structure

### `01.ethanol-transfer`
This directory contains the **dataset used for training and evaluation**.  
The data include atomic configurations, energies, and forces for the ethanol system used in the MLIP training process.

---

### `02.NN_Training.py`
This script implements the **baseline neural network architecture and the training pipeline**.  
It includes:

- Feature construction
- Neural network model definition
- Energy prediction
- Force computation via automatic differentiation
- Training loop and optimization

The implementation is written in **JAX** to leverage just-in-time compilation and efficient automatic differentiation.

---

### `03.NN_dihedral.py`
This script extends the baseline model by **introducing dihedral angle features** into the neural network input representation.

The goal is to evaluate whether adding higher-order geometric information (beyond pairwise distances and angular terms) improves the predictive accuracy and data efficiency of the MLIP.

---

### `04.TrainingCurve`
This script generates **training curves and diagnostic plots**, including:

- training loss
- validation loss
- convergence behavior

These plots help analyze model performance and training stability.

---

### `DeepH-pack`
This package is used as the **data generator** for the project.

It is based on the **deep-learning density functional theory (DL-DFT)** framework developed by our collaborator. The method enables efficient generation of ab initio–quality electronic structure data using a deep learning Hamiltonian model.

For details, please refer to:

Li, H., Wang, Z., Zou, N. et al.  
*Deep-learning density functional theory Hamiltonian for efficient ab initio electronic-structure calculation.*  
Nature Computational Science **2**, 367–377 (2022).  
https://doi.org/10.1038/s43588-022-00265-6

If you would like to use or install the DeepH package, please contact the authors of the above publication for access and further information.
---

## Dependencies

The framework relies on the following core libraries:

- JAX
- NumPy
- Matplotlib

Additional dependencies may be required depending on the training configuration.

---

## Purpose of the Project

This project explores how **compact neural architectures combined with physically informed descriptors** can improve the efficiency of machine learned interatomic potentials. The implementation is intended as a research prototype for studying MLIP design choices within the JAX ecosystem.