import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import torch
from typing import Optional, Tuple

class SnoutDataset(Dataset):
    def __init__(self, images_dir: str, labels_file: str, data_transform: Optional[str] = None, label_transform: Optional[str] = None) -> None:
        self.labels = pd.read_csv(labels_file)
        self.images = images_dir
        self.data_transform = data_transform
        self.label_transform = label_transform
    
    def __len__(self):
        return len(self.labels)
    
    def default_transformation(self, img: torch.Tensor, lbl: str) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """ Resize all images to 227 x 227. """

        # lbl is a string, separate into values
        comma = lbl.find(',')
        old_x = int(lbl[1:comma])
        old_y = int(lbl[comma+1:-1])
        
        # print(f"Old Dimensions: {(img.shape[2], img.shape[1])}, Old Label: {(old_x, old_y)}")

        # Calculate new coords based on uniform resizing ratio
        new_x = int(old_x * 227 / img.shape[2])
        new_y = int(old_y * 227 / img.shape[1])
        img = transforms.Compose([
            transforms.Resize((227, 227))
        ])(img)

        img = img.type(torch.float32) 
        img = img / 255.0 # Normalize tensor values

        return img, (new_x, new_y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Get an image and label and transform as necessary. """
        # print(self.labels.iloc[idx, 0])
        img_path = os.path.join(self.images, self.labels.iloc[idx, 0])
        img = read_image(img_path)

        # Handle images that are not 3 channel
        if img.shape[0] == 1:  
            img = img.repeat(3, 1, 1)  
        elif img.shape[0] == 4:  
            img = img[:3, :, :] 

        img, label = self.default_transformation(img, self.labels.iloc[idx, 1])
        if self.data_transform:
            img = self.img_transform(img)
        if self.label_transform:
            label = self.lbl_transform(label)
        return img, torch.Tensor(label)

    def img_transform(self, img):
        return img

    def lbl_transform(self, lbl):
        return lbl