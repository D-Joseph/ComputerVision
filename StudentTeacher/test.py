import torch
import argparse
import numpy as np
from model import ResNet18Segmentation
from torchvision.datasets import VOCSegmentation
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader

def calculate_miou(pred, target, num_classes=21):
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
    target = target.squeeze(0).cpu().numpy()
    ious = []
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection/union)
    return np.nanmean(ious)

def evaluate_model(model, dataloader, device, num_classes=21):
    total_miou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            batch_miou = calculate_miou(outputs, masks, num_classes)
            total_miou += batch_miou
            num_batches += 1
            print('Batch:', num_batches, 'Batch mIoU:', batch_miou)

    return total_miou / num_batches

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-w', metavar='path_to_weights', type=str, help='path to model weight file (.pth)', default='./outputs/weights.pth')
    args = argParser.parse_args()

    path_to_weights = args.w

    # Set working device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model with default input, output, and bottleneck sizes
    model = ResNet18Segmentation(num_classes=21)
    model.load_state_dict(torch.load(path_to_weights)) # Apply weights from training
    model.to(device=device)
    model.eval()

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    mask_transformations = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
    ])


    test_dataset = VOCSegmentation('./data', image_set='val', transform=transformations, target_transform=mask_transformations)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    miou = evaluate_model(model, test_data, device)
    print("Mean Intersection over Union (mIoU):", round(miou, 4))

if __name__ == '__main__':
    main()