import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import argparse
from model import get_model  # Make sure this function supports getting AlexNet, VGG16, and ResNet18
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from data import get_data_loaders
import os
def train(epochs: int = 30, **kwargs) -> None:
    """
    Train the model.

    Args:
        epochs (int): The number of epochs for training. Default is 30.
        kwargs: 
            - model: The model to be trained (or its string identifier for `get_model`).
            - train: The data loader for training data.
            - test: The data loader for validation data.
            - device: The device to train on ('cpu' or 'cuda').
            - fn_loss: The loss function.
            - optimizer: The optimizer to use for training.
            - scheduler: The learning rate scheduler (optional).
            - save_dir: Directory to save model outputs (optional).
    """
    print("Training Parameters:")
    for kwarg in kwargs:
        print(f"{kwarg} = {kwargs[kwarg]}")

    # Initialize model
    model = kwargs['model'] if isinstance(kwargs['model'], nn.Module) else get_model(kwargs['model'])
    model = model.to(device=kwargs['device'])  # Move model to device
    print(kwargs['model']) # print the model

    # Training
    model.train()
    losses_train = []
    losses_val = []
    start = time.time()
    save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else 'outputs'
    for epoch in range(1, epochs + 1):
        print(f"save_dir: {save_dir}") 
        model.train()
        print(f"Epoch: {epoch}/{epochs}")
        loss_train = 0.0
        for imgs, lbls in kwargs['train']:
            imgs = imgs.to(device=kwargs['device'])
            lbls = lbls.to(device=kwargs['device'])
            outputs = model(imgs)
            loss = kwargs['fn_loss'](outputs, lbls)
            kwargs['optimizer'].zero_grad()
            loss.backward()
            kwargs['optimizer'].step()
            loss_train += loss.item()

        losses_train.append(loss_train / len(kwargs['train']))

        # Validation
        model.eval()
        loss_val = 0.0
        with torch.no_grad():
            for imgs, lbls in kwargs['test']:
                imgs = imgs.to(device=kwargs['device'])
                lbls = lbls.to(device=kwargs['device'])
                outputs = model(imgs)
                loss = kwargs['fn_loss'](outputs, lbls)
                loss_val += loss.item()

        losses_val.append(loss_val / len(kwargs['test']))
        os.makedirs(kwargs['save_dir'], exist_ok=True)
        print(f"Epoch {epoch}/{epochs}: Train Loss = {loss_train:.4f}, Val Loss = {loss_val:.4f}")

        # Scheduler step
        if 'scheduler' in kwargs and kwargs['scheduler']:
            kwargs['scheduler'].step(loss_val)

        # Save model after 5 epochs and after full training
        if epoch == 5 or epoch == epochs:
            model_filename = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
            print(f"Saving model to {model_filename}")
            torch.save(model.state_dict(), model_filename)

        # Plot losses
        plt.figure(figsize=(12, 7))
        plt.plot(losses_train, label='Train Loss')
        plt.plot(losses_val, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')
        loss_plot_filename = os.path.join(save_dir, 'loss_plot.png')
        print(f"Saving loss plot to {loss_plot_filename}")
        plt.savefig(loss_plot_filename)
        plt.close()

    end = time.time()

    print(f"Training completed in {end - start:.2f} seconds")


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", type=str, default="resnet18", help="Model to train (alexnet, vgg16, resnet18)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to train on")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Directory to save model outputs")
    args = parser.parse_args()
    
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    print(f"we have parsed args: {args}")
    train_loader, test_loader = get_data_loaders(batch_size=args.batch_size, num_workers=0)

    # Initialize model
    model = get_model(args.model, 100)  # 100 classes in CIFAR-100
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
    loss_fn = nn.CrossEntropyLoss()
    
    # Train model
    train(epochs=args.epochs, model=model, train=train_loader, test=test_loader, device=args.device, fn_loss=loss_fn, optimizer=optimizer, scheduler=scheduler, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
