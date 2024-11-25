# Deep Learning for Computer Vision Projects

This repository is dedicated to exploring various applications of deep learning in computer vision. Deep learning has transformed the way we approach visual tasks, enabling models to learn from large datasets and make predictions with high accuracy. By leveraging neural networks, these projects aim to solve common problems in computer vision, such as image reconstruction, denoising, object detection, and more.

Each project in this repository demonstrates a practical implementation of deep learning models. As the repository grows, more diverse tasks and architectures will be added to showcase the flexibility and power of deep learning in solving computer vision problems.

## Projects

### 1. **MNIST Autoencoder**
   - **Description**: This project implements a 4-layer Multi-Layer Perceptron (MLP) autoencoder for the MNIST dataset. The autoencoder compresses the input images into a bottleneck layer and reconstructs them with minimal loss. The model is designed to handle noise and reconstruct clean images from noisy inputs.
   - **Features**:
     - **Autoencoding**: The model was tested by passing various images through the autoencoder. While the reconstructions generally maintain the correct digit shape, some outputs may appear slightly hazy or fuzzy.
     - **Denoising**: The model can effectively remove Gaussian noise from images, though some reconstructions may lead to digits morphing into similar-looking characters.
     - **Interpolation**: The project includes linear interpolation between the bottleneck vectors of two images, smoothly transitioning between them to visualize how the model handles morphing.
   - **Dataset**: MNIST (handwritten digits).

### 2. Pet Nose Localization
   - **Description**: This project developed a convolutional neural network model called SnoutNet to localize pet noses in images by predicting their (u,v) coordinates. The model takes in a 227x227 RGB image and outputs the predicted nose coordinates.
   - **Features**:
      - **Model Architecture**: SnoutNet has a CNN-based architecture with 3 convolutional layers followed by 3 fully connected layers. The convolutional layers extract features, while the FC layers predict the nose coordinates.
      - **Data Augmentation**: The team experimented with 3 data augmentation techniques: horizontal flip, 90-degree rotation, and a combination of both. These augmentations effectively doubled the training dataset size.
      - **Performance Evaluation**: The team evaluated the model's performance using various augmentation strategies. Flip augmentation achieved the best overall performance with the lowest mean localization error of 21.7, but introduced some high maximum errors of up to 170. Other augmentation strategies showed higher mean errors around 24-25.
   - **Dataset**: Custom pet nose dataset with labeled (u,v) coordinates.
---

### Usage
Each project folder will include the required modules in `requirements.txt`, which can be installed through `pip install -r requirements.txt`. Unique installation and usage instructions will be included in 
