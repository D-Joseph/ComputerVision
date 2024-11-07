import torch
import torch.nn as nn
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from SnoutDataset import SnoutDataset
import argparse
from model import SnoutNet
from torch.utils.data import DataLoader
from torchsummary import summary
from typing import Optional
import time

def train(epochs: Optional[int] = 30, **kwargs) -> None:
    """
    Train the model.

    Args:
        epochs (Optional[int]): The number of epochs for training. Default is 30.
        kwargs: 
            - model: The model to be trained.
            - train: The data loader for training data.
            - test: The data loader for training data.
            - device: The device to train on ('cpu' or 'cuda').
            - fn_loss: The loss function.
            - optimizer: The optimizer to use for training.
            - scheduler: The learning rate scheduler.
    """
    print("Training Parameters")
    for kwarg in kwargs:
        print(f"{kwarg} = {kwargs[kwarg]}")
    model = kwargs['model']
    
    model.train()
    losses_train = []
    losses_val = []
    start = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        for data in kwargs['train']:
            imgs, lbls = data
            imgs = imgs.to(device=kwargs['device'])
            lbls = lbls.to(device=kwargs['device'])
            outputs = model(imgs)
            loss = kwargs['fn_loss'](outputs, lbls)
            kwargs['optimizer'].zero_grad()
            loss.backward()
            kwargs['optimizer'].step()
            loss_train += loss.item()

        
        losses_train.append(loss_train/len(kwargs['train']))

        
        model.eval()  # Set the model to evaluation mode
        loss_val = 0.0
        
        with torch.no_grad():
            for imgs, lbls in kwargs['test']:
                imgs = imgs.to(device=kwargs['device'])
                lbls = lbls.to(device=kwargs['device'])
                outputs = model(imgs)
                loss = kwargs['fn_loss'](outputs, lbls)
                loss_val += loss.item()
        kwargs['scheduler'].step(loss_val)
        losses_val.append(loss_val / len(kwargs['test']))
        
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} Epoch {epoch}, Training loss {loss_train/len(kwargs['train'])}, Validation loss {loss_val / len(kwargs['test'])}")

        filename = f"./outputs_{kwargs['file_suffix']}/weights.pth"
        print(f'Saving Weights to {filename}')
        torch.save(model.state_dict(), filename)

        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        filename = f"./outputs_{kwargs['file_suffix']}/loss_plot_{kwargs['file_suffix']}.png"
        print(f'Saving to Loss Plot at {filename}')
        plt.savefig(filename)
    end = time.time()
    elapsed_time = (end - start) / 60
    print(f"Training completed in {elapsed_time:.2f} minutes")
    time_filename = f'./outputs_{kwargs["file_suffix"]}/training_time.txt'


    try:
        with open(time_filename, 'w') as f:
            f.write(f"Training completed in {elapsed_time:.2f} minutes\n") 
            f.write(f"Final loss: {loss.item()}\n")
            f.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} "
                f"Epoch {epoch}, "
                f"Training loss {loss_train/len(kwargs['train']):.4f}, "
                f"Validation loss {loss_val/len(kwargs['test']):.4f}\n"
            )
    except IOError as e:
        print(f"An error occurred while writing to the file: {e}")


def init_weights(m):
    """ Initialize default weights for the model. """
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-t', metavar='transformation', type=str, choices=['f', 'r', 'fr'], help='one of f, r, or fr to expand the dataset through a flip (f), rotation (r), or both (fr).')
    argParser.add_argument('-i', metavar='images_directory', type=str, help='path to images directory, defaults to ./oxford-iiit-pet-noses/images-original/images/', default='./oxford-iiit-pet-noses/images-original/images/')
    argParser.add_argument('-l', metavar='labels', type=str, help='path to labels directory, defaults to ./oxford-iiit-pet-noses/', default='./oxford-iiit-pet-noses/')
    argParser.add_argument('-b', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    args = argParser.parse_args()

    transformation = []
    if args.t and 'f' in args.t:
        transformation.append('flip')
    if args.t and 'r' in args.t:
        transformation.append('rotate')
    transformation_str = '_'.join(transformation) if transformation else 'none'

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    model = SnoutNet()
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)
    train_data = DataLoader(
        SnoutDataset(args.i, f"{args.l}/train_noses.txt", transform=transformation),
        batch_size=args.b, 
        shuffle=True,
        collate_fn=SnoutDataset.remove_failed
    )

    test_data = DataLoader(
        SnoutDataset(args.i, f"{args.l}/test_noses.txt"),
        batch_size=args.b,
        shuffle=True,
        collate_fn=SnoutDataset.remove_failed
    )
    # Ensure that output folder is ready
    os.makedirs(f'./outputs_{transformation_str}', exist_ok=True)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    train(
            optimizer=optimizer,
            model=model,
            fn_loss=loss_fn,
            train=train_data,
            test=test_data,
            scheduler=scheduler,
            device=device,
            file_suffix = transformation_str
            )


if __name__ == '__main__':
    main()