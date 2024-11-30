import torch
import numpy as np
from model import ResNet18Segmentation
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from torchvision.models.segmentation import fcn_resnet50

def visualize_segmentation(image, mask18, mask50, ground_truth, class_colors, alpha=0.5):
    """
    Visualizes input image, predicted mask, ground truth mask, and overlay.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W) in range [0, 1].
        mask18 (torch.Tensor): The predicted segmentation mask tensor (H, W) for the ResNet18 model.
        mask50 (torch.Tensor): The predicted segmentation mask tensor (H, W) for the ResNet50 model.
        ground_truth (torch.Tensor): The ground truth segmentation mask (H, W).
        class_colors (list): List of RGB tuples for each class.
        alpha (float): Transparency of the overlay.
    """
    # Convert image and masks to numpy arrays
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_np18 = mask18.cpu().numpy()
    mask_np50 = mask50.cpu().numpy()
    ground_truth_np = ground_truth.cpu().numpy()

    # Create blank RGB images for the masks
    mask_rgb18 = np.zeros((*mask_np18.shape, 3), dtype=np.uint8)
    mask_rgb50 = np.zeros((*mask_np50.shape, 3), dtype=np.uint8)
    gt_rgb = np.zeros((*ground_truth_np.shape, 3), dtype=np.uint8)

    for cls, color in enumerate(class_colors):
        mask_rgb18[mask_np18 == cls] = color
        mask_rgb50[mask_np50 == cls] = color
        gt_rgb[ground_truth_np == cls] = color

    # Overlay the predicted mask on the input image
    overlay18 = (alpha * mask_rgb18 + (1 - alpha) * image_np * 255).astype(np.uint8)
    overlay50 = (alpha * mask_rgb50 + (1 - alpha) * image_np * 255).astype(np.uint8)

    # Plot results
    plt.figure(figsize=(16, 8))
    plt.subplot(2, 3, 1)
    plt.title("Input Image")
    plt.imshow(image_np)
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.title("Ground Truth")
    plt.imshow(gt_rgb)
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.title("ResNet18 Predicted Mask")
    plt.imshow(mask_rgb18)
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.title("ResNet50 Predicted Mask")
    plt.imshow(mask_rgb50)
    plt.axis("off")
    
    plt.subplot(2, 3, 5)
    plt.title("ResNet18 Overlay")
    plt.imshow(overlay18)
    plt.axis("off")    
    
    plt.subplot(2, 3, 6)
    plt.title("ResNet50 Overlay")
    plt.imshow(overlay50)
    plt.axis("off")

    plt.tight_layout()
    plt.show()
    plt.savefig('./outputs/visualization_rn50.jpg')

def main():
    # Define colors for the segmentation classes (Example for 21 classes)
    class_colors = [
        (0, 0, 0),       # Background
        (128, 0, 0),     # Class 1
        (0, 128, 0),     # Class 2
        (128, 128, 128)  # Class 20
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained model
    model18 = ResNet18Segmentation(num_classes=21)
    model18.load_state_dict(torch.load('./outputs/weights.pth'))
    model18.to(device)
    model18.eval()
    model50 = fcn_resnet50(pretrained=True)
    model50.to(device)
    model50.eval()

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
    image, ground_truth = test_dataset[0] 
    image, ground_truth = image.to(device), ground_truth.to(device)

    # Perform inference
    with torch.no_grad():
        output50 = model50(image.unsqueeze(0))['out']  # Add batch dimension
        prediction50 = torch.argmax(output50, dim=1).squeeze(0)  # Convert logits to class indices
        output18 = model18(image.unsqueeze(0))  # Add batch dimension
        prediction18 = torch.argmax(output18, dim=1).squeeze(0)  # Convert logits to class indices

    # Visualize the results
    visualize_segmentation(image, prediction18, prediction50, ground_truth, class_colors)


if __name__ == '__main__':
    main()
