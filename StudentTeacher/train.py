import torch
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from torch.utils.data import DataLoader
from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_pil_image
import numpy as np
from PIL import Image

# Step 1: Load Pretrained Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = fcn_resnet50(pretrained=True)
model = model.to(device)
model.eval()

# Step 2: Define Dataset and Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

def preprocess_mask(mask):
    # Convert target to indices compatible with FCN output
    mask = mask.resize((256, 256), Image.NEAREST)
    mask = torch.from_numpy(np.array(mask, dtype=np.int64))
    return mask

class VOCSegmentationWithTransforms(VOCSegmentation):
    def __getitem__(self, index):
        img, mask = super().__getitem__(index)
        img = transform(img)
        mask = preprocess_mask(mask)
        return img, mask

# Download and prepare the VOCSegmentation dataset
val_dataset = VOCSegmentationWithTransforms(
    root="data", year="2012", image_set="val", download=True
)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Step 3: Define mIoU Metric
def compute_miou(pred, target, num_classes=21):
    pred = pred.argmax(dim=1)
    iou_per_class = []
    for cls in range(num_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().item()
        union = (pred_cls | target_cls).sum().item()
        if union > 0:
            iou_per_class.append(intersection / union)
    return np.mean(iou_per_class)

# Step 4: Evaluate Model
def evaluate_model(model, dataloader, num_classes=21):
    total_miou = 0.0
    num_batches = 0

    with torch.no_grad():
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            batch_miou = compute_miou(outputs, masks, num_classes)
            total_miou += batch_miou
            num_batches += 1

    return total_miou / num_batches

# Run Evaluation
miou = evaluate_model(model, val_loader)
print(f"Mean Intersection over Union (mIoU): {miou:.4f}")
