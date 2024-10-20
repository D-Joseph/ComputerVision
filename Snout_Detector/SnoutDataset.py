import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class SnoutDataset(Dataset):
    def __init__(self, images_dir, labels_file, data_transform=False, label_transform=False):
        self.labels = pd.read_csv(labels_file)
        self.images = images_dir
        self.data_transform = data_transform
        self.label_transform = label_transform
    
    def __len__(self):
        return len(self.labels)
    
    def default_transformation(self, img, lbl):
        """ Resize all images to 227 x 227. """

        # lbl is a string, separate into values
        comma = lbl.find(',')
        old_x = int(lbl[1:comma])
        old_y = int(lbl[comma+1:-1])
        
        print(f"Old Dimensions: {(img.shape[2], img.shape[1])}, Old Label: {(old_x, old_y)} - {lbl}")

        # Calculate new coords based on uniform resizing ratio
        new_x = int(old_x * 227 / img.shape[2])
        new_y = int(old_y * 227 / img.shape[1])
        img = transforms.Compose([
            transforms.Resize((227, 227))
        ])(img)

        print(f"New Dimensions: {(img.shape[2], img.shape[1])}, New Label: {(new_x, new_y)}")

        return img, (new_x, new_y)

    def __getitem__(self, idx):
        """ Get an image and label and transform as necessary. """
        print(self.labels.iloc[idx, 0])
        img_path = os.path.join(self.images, self.labels.iloc[idx, 0])
        image, label = self.default_transformation(read_image(img_path), self.labels.iloc[idx, 1])
        if self.data_transform:
            image = self.img_transform(image)
        if self.label_transform:
            label = self.lbl_transform(label)
        return image, label

    def img_transform(self, img):
        return img

    def lbl_transform(self, lbl):
        return lbl