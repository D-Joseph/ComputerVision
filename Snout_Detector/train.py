import torch
import torch.nn as nn
import os
import torch.optim as optim
import matplotlib.pyplot as plt
from SnoutDataset import SnoutDataset
import datetime
import argparse
from model import SnoutNet
from torch.utils.data import DataLoader
from torchsummary import summary
from typing import Optional

def train(epochs: Optional[int] = 30, **kwargs) -> None:
    """
    Train the model.

    Args:
        epochs (Optional[int]): The number of epochs for training. Default is 30.
        kwargs: 
            - model: The model to be trained.
            - loader: The data loader for training data.
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
    os.makedirs('./outputs', exist_ok=True)
    losses_train = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch}")
        loss_train = 0.0
        for data in kwargs['loader']:
            imgs, lbls = data
            imgs = imgs.to(device=kwargs['device'])
            lbls = lbls.to(device=kwargs['device'])
            outputs = model(imgs)
            loss = kwargs['fn_loss'](outputs, lbls)
            kwargs['optimizer'].zero_grad()
            loss.backward()
            kwargs['optimizer'].step()
            loss_train += loss.item()

        kwargs['scheduler'].step(loss_train)
        losses_train += [loss_train/len(kwargs['loader'])]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, loss_train/len(kwargs['loader'])))

        print('Saving Weights to ./weights.pth')
        torch.save(model.state_dict(), './weights.pth')

        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        print('Saving to Loss Plot at ./outputs/loss_plot.png')
        plt.savefig('./outputs/loss_plot.png')
        return

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-t', metavar='transformation', type=str, help='One of f, r, or fr to expand dataset through a flip, rotation, or both.')
    args = argParser.parse_args()
    
    transformation = []
    if args.t.find('f') > -1:
        transformation.append('flip')
    if args.t.find('r') > -1:
        transformation.append('rotate')
    
    print(transformation)

    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    model = SnoutNet()
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)
    print(args.t)
    dataloader = DataLoader(
        SnoutDataset('./oxford-iiit-pet-noses/images-original/images', './oxford-iiit-pet-noses/train_noses.txt', transform=args.t),
        batch_size=64, 
        shuffle=True
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = nn.MSELoss(size_average=None, reduce=None, reduction='mean')

    train(
            optimizer=optimizer,
            model=model,
            fn_loss=loss_fn,
            loader=dataloader,
            scheduler=scheduler,
            device=device,
            )


if __name__ == '__main__':
    main()