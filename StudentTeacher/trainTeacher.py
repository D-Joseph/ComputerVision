import torch
import argparse
from typing import Optional
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torchvision.models.segmentation import fcn_resnet50
import numpy as np
import random
from augmentations import get_train_augmentations, get_val_augmentations
from segmentationDataset import SegmentationDataset

def train(epochs: Optional[int] = 30, **kwargs) -> None:
    print("Training Parameters")
    for kwarg in kwargs:
        print(f"{kwarg} = {kwargs[kwarg]}")

    model = kwargs['model']
    model.train()
    optimizer = kwargs['optimizer']
    scheduler = kwargs['scheduler']
    loss_function = kwargs['loss_function']

    losses_train = []  # Array to store training losses for plotting
    losses_val = []    # Array to store validation losses for plotting

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        val_loss = 0

        # Training Loop
        for data, labels in kwargs['train']:
            data, labels = data.to(kwargs['device']), labels.to(kwargs['device']).long()            
            optimizer.zero_grad()
            outputs = model(data)['out']
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(kwargs['train'])
        losses_train.append(train_loss)

        # Validation Loop
        model.eval()
        with torch.no_grad():
            for data, labels in kwargs['test']:
                data, labels = data.to(kwargs['device']), labels.to(kwargs['device']).long()                
                outputs = model(data)['out']
                val_loss += loss_function(outputs, labels).item()

        val_loss /= len(kwargs['test'])
        losses_val.append(val_loss)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        # Save Model Weights


        # Save Loss Plot
        plt.figure()
        plt.plot(losses_train, label="Training Loss")
        plt.plot(losses_val, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(kwargs['output'], "loss_plot.png"))
        plt.close()
    save_path = os.path.join(kwargs['output'], f"weights_teacher.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Saved model weights to {save_path}")

def main():
    # Argument Parser
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-e", type=int, default=30, help="Number of epochs")
    argParser.add_argument("-b", type=int, default=8, help="Batch size")
    argParser.add_argument("-o", type=str, default="./output/fcn_resnet50", help="Output directory")
    argParser.add_argument("-d", type=str, default="./data", help="Base dataset directory")
    args = argParser.parse_args()



    # Make Sure Output Folder Exists
    if not os.path.exists(args.o):
        os.makedirs(args.o)

    # Check Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define Transformations
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    target_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.Lambda(lambda x: torch.tensor(np.array(x), dtype=torch.long)),
    ])


    train_images_dir = f"{args.d}/VOCdevkit/VOC2012/JPEGImages"
    train_masks_dir = f"{args.d}/VOCdevkit/VOC2012/SegmentationClass"
    train_file_list = f"{args.d}/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt"

    val_images_dir = f"{args.d}/VOCdevkit/VOC2012/JPEGImages"
    val_masks_dir = f"{args.d}/VOCdevkit/VOC2012/SegmentationClass"
    val_file_list = f"{args.d}/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt"

    train_transform = get_train_augmentations()
    val_transform = get_val_augmentations()

    train_dataset = SegmentationDataset(
        image_root=train_images_dir,
        mask_root=train_masks_dir,
        file_list=train_file_list,
        transform=train_transform
    )

    val_dataset = SegmentationDataset(
        image_root=val_images_dir,
        mask_root=val_masks_dir,
        file_list=val_file_list,
        transform=val_transform
    )
    train_loader = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.b, shuffle=False, num_workers=4, pin_memory=True)

    # Load Pretrained Model
    model = fcn_resnet50(weights="DEFAULT")
    model = model.to(device)

    # Define Optimizer, Scheduler, and Loss Function
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, verbose=True)
    loss_function = nn.CrossEntropyLoss(ignore_index=255)  # Ignore padding in VOC dataset

    # Call Train Function
    train(
        epochs=args.e,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_function=loss_function,
        train=train_loader,
        test=val_loader,
        device=device,
        output=args.o,
    )


if __name__ == "__main__":
    main()
