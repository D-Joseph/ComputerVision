import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from SnoutDataset import SnoutDataset
import datetime
import argparse
from model import SnoutNet
from torch.utils.data import DataLoader
from torchsummary import summary

#   some default parameters, which can be overwritten by command line arguments
save_file = 'weights.pth'
n_epochs = 30
batch_size = 256
bottleneck_size = 32
plot_file = 'plot.png'

def train(epochs=30, **kwargs):
    """ Train the model.

    kwargs:
    - model
    - epochs
    - loader
    - device
    - fn_loss
    - optimizer
    - scheduler
    """
    print("Training Parameters")
    for kwarg in kwargs:
        print(f"{kwarg} = {kwargs[kwarg]}")
    model = kwargs['model']
    
    model.train()

    losses_train = []
    for epoch in range(epochs):
        print(f"**********\n\nEpoch: {epoch}\n\n**********")
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

        print('Saving Weights to ./outputs/weights.pth')
        torch.save(model.state_dict(), './outputs/weights.pth')

        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        print('Saving Loss Plot to ./outputs/loss_plot.png')
        plt.savefig('./outputs/loss_plot.png')

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def main():
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print(f'Using: {device}')

    model = SnoutNet()
    model.to(device)
    model.apply(init_weights)
    summary(model, model.input_shape)

    dataloader = DataLoader(
        SnoutDataset('./oxford-iiit-pet-noses/images-original/images', './oxford-iiit-pet-noses/train_noses.txt'),
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