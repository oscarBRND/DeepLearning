# Deep Learning Project

## Description

This project explores and implements deep learning algorithms progressively, starting from **low-level, from-scratch implementations** to build strong theoretical and practical foundations, and gradually moving toward **deeper and more advanced architectures**.

The main objective is to understand how neural networks operate internally (forward pass, backpropagation, optimization dynamics), while preparing a clean transition toward modern deep learning practices and libraries.

---

## Implemented Components

### 1. Binary Classifier (Single Neuron)
A minimal binary classifier implemented manually with NumPy to understand:

- Linear decision boundaries  
- Sigmoid activation  
- Binary cross-entropy loss  
- Gradient descent optimization  

---

### 2. Two-Layer Neural Network (2 → nₕ → 1)
A fully manual neural network implementation including:

- Forward propagation  
- Backpropagation  
- Gradient descent training  
- Loss and accuracy tracking  
- Decision boundary visualization  

#### Architecture
<p align="center">
  <img src="src/architecture.png" width="450">
</p>

#### Training Animation
![Training Animation](src/nn_training_contour.gif)

---

### 3. Generalized Deep Neural Network (n Layers)
The network has been **generalized to support arbitrary depth**:

- Configurable architecture: `[n₀, n₁, ..., n_L]`
- Modular forward and backward passes
- Clean separation between:
  - initialization
  - forward propagation
  - backpropagation
  - optimization

#### Supported Activation Functions
The implementation now supports multiple activation functions:

- **Sigmoid**
- **Tanh**
- **ReLU**

Each activation is handled consistently in both the forward and backward passes, enabling experimentation with different architectures and learning behaviors.

---

## Training & Optimization

Current optimization method:
- **Batch Gradient Descent**

Tracked metrics:
- Binary cross-entropy (NLL)
- Accuracy
- Training dynamics through animated decision boundaries

---

## Next Steps

### Cleaner & Object-Oriented Design
Refactor the current functional implementation into a **clean, extensible OOP design**:

- `Layer` abstraction  
- `NeuralNetwork` class  
- Plug-and-play activations and optimizers  

---

### Optimization Improvements
Implement more advanced optimizers:

- Momentum-based Gradient Descent  
- RMSProp  
- Adam  

---

### Computer Vision (CNNs)
Introduce convolutional neural networks using a deep learning framework (PyTorch or Keras):

- Basic CNN for image classification  
- Visualization of learned filters and feature maps  

---

### Future Explorations
- Batch Normalization  
- Residual Networks (ResNet)  
- Deeper architectures & training stability  
- Regularization techniques  

---

## Goal

The goal of this project is to develop **a strong, practical understanding of deep learning from first principles**, observe training dynamics in detail, and progressively transition toward **industrial-grade architectures and workflows**.

This repository serves both as a learning tool and a technical showcase of low-level deep learning implementations.
