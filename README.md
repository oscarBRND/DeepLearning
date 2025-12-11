# Deep Learning Project

## Description

This project aims to explore and implement a wide range of deep learning algorithms, starting with manual low-level implementations to build strong foundations, then progressively moving toward more advanced and higher-level models.  
The objective is to better understand how neural networks work internally while also gaining experience with modern deep learning techniques and architectures.

---

### 1. Binary Classifier (Single Neuron)
A minimal binary classifier coded manually using NumPy to understand fundamental classification principles.

### 2. Two-Layer Neural Network (2 → nₕ → 1)
A complete neural network implementation including:

- Forward propagation  
- Backpropagation  
- Gradient descent training  
- Loss and accuracy tracking  
- Visualization of the decision boundary  

#### Architecture 
<p align="center">
  <img src="src/architecture.png" width="450">
</p>

#### Training Animation
![Training Animation](src/nn_training_contour.gif)

---
## Next Step

### Neural Network Generalization
Extend the implementation to support **deep architectures (n layers)**:

- Modular forward/backward passes  
- Multiple activation functions  
- Cleaner, object-oriented design  

### Optimization Improvements
Implement more powerful optimizers:

- Adam  
- RMSProp  
- Momentum-based GD  

### Computer Vision (CNNs)
Introduce convolutional architectures using a deep learning library (Keras or PyTorch):

- Basic CNN for image classification  
- Visualization of feature maps and learned filters  

### Future Explorations: 
-ResNet
-Batch normalization
-To be determined as the project evolves

## Goal

The aim is to gain hands-on experience by implementing essential deep learning components, observing training behavior, and gradually transitioning toward more complex neural architectures and real-world applications.


