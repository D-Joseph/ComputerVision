import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from torch.utils.data import Dataset
import torch
import os

class SegmentationDataset(Dataset):
    def __init__(self, image_root, mask_root, file_list, transform=None, num_classes=21, ignore_index=255):
        """
        Args:
            image_root (str): Directory containing input images.
            mask_root (str): Directory containing segmentation masks.
            file_list (str): Path to the file listing image names (without extensions).
            transform (callable, optional): Albumentations transformation pipeline.
            num_classes (int, optional): Total number of classes.
            ignore_index (int, optional): Value to ignore during training (e.g., 255).
        """
        self.image_root = image_root
        self.mask_root = mask_root
        self.file_list = file_list
        self.transform = transform
        self.num_classes = num_classes
        self.ignore_index = ignore_index

        self.image_paths = []
        self.mask_paths = []

        # Read the file list and generate full paths
        with open(self.file_list, "r") as f:
            for line in f:
                file_name = line.strip()
                self.image_paths.append(os.path.join(self.image_root, f"{file_name}.jpg"))
                self.mask_paths.append(os.path.join(self.mask_root, f"{file_name}.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image and mask
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at path: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found at path: {mask_path}")

        # Ensure mask values are valid
        mask[mask >= self.num_classes] = self.ignore_index  # Ignore invalid class indices

        # Apply transformations using Albumentations
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        # Convert mask to LongTensor for CrossEntropyLoss
        mask = torch.tensor(mask, dtype=torch.long)

        return image, mask
