import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms import InterpolationMode
import numpy as np

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
            outputs = model(images)['out']
            batch_miou = calculate_miou(outputs, masks, num_classes)
            total_miou += batch_miou
            num_batches += 1
            print('Batch:', num_batches, 'Batch mIoU:', batch_miou)

    return total_miou / num_batches

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    model = fcn_resnet50(pretrained=True)
    model = model.to(device)
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
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    miou = evaluate_model(model, test_loader, device)
    print("Mean Intersection over Union (mIoU):", round(miou, 4))

if __name__ == "__main__":
    main()