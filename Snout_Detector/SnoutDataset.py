import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch
from typing import Optional, Tuple, Literal, List

class SnoutDataset(Dataset):
    def __init__(self, images_dir: str, labels_file: str, transform: Optional[List[Literal['flip', 'rotate']]] = []) -> None:
        self.labels = pd.read_csv(labels_file)
        self.images = images_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.labels) * 2 if self.transform else len(self.labels)
    
    def default_transformation(self, img: torch.Tensor, lbl: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """ Resize all images to 227 x 227 and convert tuple to int from str. """

        # lbl is a string, separate into values
        comma = lbl.find(',')
        old_x = int(lbl[1:comma])
        old_y = int(lbl[comma+1:-1])
        
        print(f"Old Dimensions: {(img.shape[2], img.shape[1])}, Old Label: {(old_x, old_y)}")

        # Calculate new coords based on uniform resizing ratio
        new_x = int(old_x * 227 / img.shape[2])
        new_y = int(old_y * 227 / img.shape[1])
        img = transforms.Compose([
            transforms.Resize((227, 227))
        ])(img)

        img = img.type(torch.float32) 
        img = img / 255 # Normalize tensor values

        return img, (new_x, new_y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get an image and label and transform as necessary. """
        # To handle augmentation, we will first add the original images into the dataset, and then wrap around for each augmentation
        true_idx = idx % len(self.labels)
        # print(idx, self.labels.iloc[true_idx, 0])
        img_path = os.path.join(self.images, self.labels.iloc[true_idx, 0])
        img = read_image(img_path)

        # Handle images that are not 3 channel
        if img.shape[0] == 1:  
            img = img.repeat(3, 1, 1)  
        elif img.shape[0] == 4:  
            img = img[:3, :, :] 

        img, label = self.default_transformation(img, self.labels.iloc[true_idx, 1])
        # print(f"Resized Dimensions: {(img.shape[2], img.shape[1])}, Resized Label: {(label[0], label[1])}")

        # Second half of indices should have their image augmented
        if idx >= len(self.labels):
            img, label = self.optional_transformations(img, label)
            
        # print(f"New Dimensions: {(img.shape[2], img.shape[1])}, New Label: {(label[0], label[1])}")
        return img, torch.Tensor(label)

    def optional_transformations(self, img: torch.Tensor, label: Tuple[int, int]) -> Tuple[torch.Tensor, Tuple[int, int]]:
        for t in self.transform:
            # Perform a horizontal flip
            if t == 'flip':
                img = transforms.functional.hflip(img)
                label = (227 - label[0], label[1]) 

            # Rotate the image by 90 degrees
            elif t == 'rotate':
                img = transforms.functional.rotate(img, 90)
                label = (label[1], 226 - label[0])
        return img, label
