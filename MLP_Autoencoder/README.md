# Model Details
This project implements a multi-layer perceptron (MLP) autoencoder to process and reconstruct handwritten digits from the MNIST dataset. The model architecture is as follows:

Input: 28x28px image (preprocessed into a 1x784 tensor)
Encoder:
- FC Layer 1: 784 -> 392 (ReLU activation)
- FC Layer 2 (Bottleneck): 392 -> 8 (ReLU activation)

Decoder:
- FC Layer 3: 8 -> 392 (ReLU activation)
- FC Layer 4 (Output): 392 -> 784 (Sigmoid activation)

The model compresses the input image into a small bottleneck representation and then reconstructs it to its original size. ReLU activation introduces non-linearity, while the final sigmoid function models pixel intensity in the reconstructed image.
Show Image
Figure 1: Diagrammatic representation of the developed model
# Training Details
The MLP autoencoder was trained using PyTorch with the following specifications:
- Dataset: MNIST training set
- Initialization: Xavier uniform (weights), 0.01 (bias)
- Optimizer: Adam
- Learning rate: 1e-3
- Weight decay: 1e-5
- Scheduler: ReduceLROnPlateau (mode: min)
- Loss Function: Mean Squared Error (MSE)
- Batch Size: 2048 images
- Epochs: 50
- Activation Function: ReLU

# Results
## Autoencoding
The model demonstrates good reconstruction capabilities, generally maintaining the correct discernible shape of the input digits with varying levels of haziness.

## Image Denoising
The model's performance in denoising is inconsistent. While it handles noise well in some cases, there are instances where the reconstructed digit is misinterpreted (e.g., a 2 reconstructed as an 8).

## Bottleneck Interpolation
The model excels at interpolating between two encoded representations, demonstrating a clear morphing between different digits.

## Loss Plot
The loss plot indicates successful training without over or underfitting, showing effective weight adjustments over the 50 training epochs.

# Usage
The lab1.py script provides the following functionalities:
1. Autoencoding: Reconstruct a selected MNIST image
2. Image Denoising: Add Gaussian noise to an image and attempt reconstruction
3. Bottleneck Interpolation: Linearly interpolate between two encoded representations
Follow the prompts in the script to test these functionalities.
