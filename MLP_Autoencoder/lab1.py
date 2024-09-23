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

def output_plots(imgs, filename="Output Image"):
    """ Create dynamically sized plots for varying len(imgs). """
    f = plt.figure()
    for idx, img in enumerate(imgs):
        f.add_subplot(1,len(imgs), idx + 1)
        plt.imshow(img.view(28, 28), cmap='gray') # Reconstruct image to 28x28
    filename = 'outputs/' + filename
    plt.savefig(filename)
    print(f'Saved Image to "{filename}"')
    plt.show()

def index_test(idx, dataset, model, device):
    """ Test the autoencoder on a given MNIST image. """
    img = image_manipulation(dataset.data[idx]).to(device=device)
    img.to(device=device)
    with torch.no_grad():
        output = model(img)
    output_plots([img, output], filename=f'Autoencoding - Index {idx}.png')

def add_noise(idx, dataset, model, device):
    """ Add uniformly distributed noise to the images and test the autoencoder. """
    orig = image_manipulation(dataset.data[idx]).to(device=device)
    orig.to(device=device)
    noisy = orig + (torch.rand(orig.size()))
    with torch.no_grad():
        output = model(noisy)
    output_plots([orig, noisy, output], filename=f'Denoising - Index {idx}.png', )

def linear_interpolation(i1, i2, dataset, model, device, num_steps=8):
    """ In steps, merge the autoencoding output of 2 images together. """
    img1 = image_manipulation(dataset.data[i1]).to(device=device)
    img2 = image_manipulation(dataset.data[i2]).to(device=device)
    interpols = [img1]
    with torch.no_grad():
        enc1 = model.encode(img1)
        enc2 = model.encode(img2)
        for i in range(1, num_steps + 1):
            # The encoding weight should go from the first image to the second image in uniform steps
            interpolation = enc1 *(1 - (i/num_steps)) + enc2 * (i/num_steps)
            output = model.decode(interpolation)
            interpols.append(output)
    interpols.append(img2)
    output_plots(interpols, filename=f'Interpolating - Indexes {i1} and {i2}.png')

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

    while True:
        idx = int(input("Test the autoencoder. Select an index: "))
        if 0 <= idx < len(test_dataset):
            index_test(idx, test_dataset, model, device)
            add_noise(idx, test_dataset, model, device)
            idx2 = -1
            while not (0 <= idx2 < len(test_dataset)):
                  idx2 = int(input(f"Select another index for linear interpolation (max {len(test_dataset)} -1): "))
            linear_interpolation(idx, idx2, test_dataset, model, device) 
        else:
            print(f"Index out of range. Max = {len(test_dataset) - 1}")

if __name__ == "__main__":
    main() 