import matplotlib.pyplot as plt
from data import get_data_loaders
import torch

def show_images(loader):
    data_iter = iter(loader)
    images, labels = next(data_iter)

    # Print the shape of the batch
    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    # Display the first image in the batch
    image = images[0]  # Get the first image in the batch
    label = labels[0]  # Get the corresponding label

    # Convert the tensor image to a NumPy array and denormalize for visualization
    image = image.permute(1, 2, 0)  # Change from (C, H, W) to (H, W, C)
    image = image * torch.tensor([0.2675, 0.2565, 0.2761]) + torch.tensor([0.5071, 0.4867, 0.4408])  # Denormalize
    image = image.numpy()

    # Plot the image
    plt.imshow(image)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()
    
def main():
    train_loader, test_loader = get_data_loaders()
    show_images(train_loader)
    show_images(test_loader)

    
if __name__ == "__main__":
    main()