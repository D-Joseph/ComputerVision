import torch
import numpy as np
from model import ResNet18Segmentation
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt


def visualize_segmentation(image, mask, ground_truth, class_colors, alpha=0.5):
    """
    Visualizes input image, predicted mask, ground truth mask, and overlay.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W) in range [0, 1].
        mask (torch.Tensor): The predicted segmentation mask tensor (H, W).
        ground_truth (torch.Tensor): The ground truth segmentation mask (H, W).
        class_colors (list): List of RGB tuples for each class.
        alpha (float): Transparency of the overlay.
    """
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy()
    ground_truth_np = ground_truth.cpu().numpy()

    # Create blank RGB images for the masks
    mask_rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    gt_rgb = np.zeros((*ground_truth_np.shape, 3), dtype=np.uint8)

    for cls, color in enumerate(class_colors):
        mask_rgb[mask_np == cls] = color
        gt_rgb[ground_truth_np == cls] = color

    # Overlay the predicted mask on the input image
    overlay = (alpha * mask_rgb + (1 - alpha) * image_np * 255).astype(np.uint8)

    # Plot results
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 4, 1)
    plt.title("Input Image")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.title("Ground Truth")
    plt.imshow(gt_rgb)
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.title("Predicted Mask")
    plt.imshow(mask_rgb)
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.title("Overlay")
    plt.imshow(overlay)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig('./outputs/visualization.jpg')

def main():
    # Define colors for the segmentation classes (Example for 21 classes)
    class_colors = [
        (0, 0, 0),       # Background
        (128, 0, 0),     # Class 1
        (0, 128, 0),     # Class 2
        # Add more colors as needed for each class...
        (128, 128, 128)  # Class 20
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model = ResNet18Segmentation(num_classes=21)
    model.load_state_dict(torch.load('./outputs/weights.pth'))
    model.to(device)
    model.eval()

    # Define image transformations
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    mask_transformations = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
    ])

    # Load a single image and mask
    test_dataset = VOCSegmentation('./data', image_set='val', transform=transformations, target_transform=mask_transformations)
    image, ground_truth = test_dataset[0]  # Load the first image and ground truth
    image, ground_truth = image.to(device), ground_truth.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(image.unsqueeze(0))  # Add batch dimension
        prediction = torch.argmax(output, dim=1).squeeze(0)  # Convert logits to class indices

    # Visualize the results
    visualize_segmentation(image, prediction, ground_truth, class_colors)


if __name__ == '__main__':
    main()
