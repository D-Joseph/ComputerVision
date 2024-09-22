from model import autoencoderMLP4Layer
import torch
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse


def image_manipulation(image):
    """ Normalize and flatten the image. """
    image = image.type(torch.float32) 
    image = (image - torch.min(image)) / torch.max(image) # Normalize tensor values
    image = image.view(1, image.shape[0]*image.shape[1]).type(torch.FloatTensor) # Convert 2D image to 1D (1x(28*28))
    return image

def index_test(idx, dataset, model, device):
    """ Test the autoencoder on a given MNIST image. """
    img = image_manipulation(dataset.data[idx]).to(device=device)
    img.to(device=device)
    with torch.no_grad():
        output = model(img)

    # Display original and reconstruction in plot
    f = plt.figure()
    f.add_subplot(1,2,1)
    plt.imshow(img.view(28, 28), cmap='gray') # Reconstruct image to 28x28
    f.add_subplot(1,2,2)
    plt.imshow(output.view(28, 28), cmap='gray')
    filename = f'./outputs/Autoencoder Test - Index {idx}.png'
    plt.savefig(filename)
    print(f'Saved autoencoder test to "{filename}"')
    plt.show()

def add_noise(idx, dataset, model, device):
    orig = image_manipulation(dataset.data[idx]).to(device=device)
    orig.to(device=device)
    noisy = orig + (torch.rand(orig.size()))
    with torch.no_grad():
        output = model(noisy)
    
    
    f = plt.figure()
    f.add_subplot(1, 3, 1)
    plt.imshow(orig.view(28, 28).cpu(), cmap='gray')  # Noisy image
    f.add_subplot(1, 3, 2)
    plt.imshow(noisy.view(28, 28).cpu(), cmap='gray')  # Reconstructed image
    f.add_subplot(1, 3, 3)
    plt.imshow(output.view(28, 28).cpu(), cmap='gray')  # Reconstructed image
    filename = f'./outputs/Denoising Test - Index {idx}.png'
    plt.savefig(filename)
    print(f'Saved denoising test to "{filename}"')
    plt.show()


def main():
    # Command line argument for specifying path to parameters
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-l', metavar='path_to_weights', type=str, help='path to model weight file (.pth)', default='MLP.8.pth')
    args = argParser.parse_args()

    path_to_weights = args.l

    # Set working device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with default input, output, and bottleneck sizes
    model = autoencoderMLP4Layer()
    model.load_state_dict(torch.load(path_to_weights)) # Apply weights from training
    model.to(device=device)
    model.eval()

    # Convert MNIST images to Tensors
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = MNIST('./data/mnist', train=False, download=True, transform=test_transform)
    

    idx = 0 
    while idx >= 0:
        idx = int(input("Test the autoencoder. Select an index: "))
        if 0 <= idx < len(test_dataset):
            index_test(idx, test_dataset, model, device)
            add_noise(idx, test_dataset, model, device)
        else:
            print(f"Index out of range. Max = {len(test_dataset) - 1}")


if __name__ == "__main__":
    main() 