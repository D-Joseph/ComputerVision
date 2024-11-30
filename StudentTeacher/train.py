import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import time
from torchvision.datasets import VOCSegmentation
from torch.nn import CrossEntropyLoss
from torchvision.transforms import InterpolationMode
from model import ResNet18Segmentation 
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

def train(epochs = 30, **kwargs) -> None:
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
        print(kwarg, "=", kwargs[kwarg])
    model = kwargs['model']
    
    model.train()
    losses_train = []
    losses_val = []
    start = time.time()

    for epoch in range(1, epochs+1):
        model.train()
        print("Epoch:", epoch)
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
        
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "Epoch:", epoch, "Training loss:", loss_train/len(kwargs['train']), "Validation loss:", loss_val / len(kwargs['test']))

        filename = "./outputs/weights.pth"
        print("Saving Weights to", filename)
        torch.save(model.state_dict(), filename)

        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        filename = "./outputs/loss_plot.png"
        print('Saving to Loss Plot at', filename)
        plt.savefig(filename)
    end = time.time()
    elapsed_time = (end - start) / 60
    print("Training completed in", round(elapsed_time, 2), "minutes")
    time_filename = './outputs/training_time.txt'


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Segmentation(num_classes=21)

    model = model.to(device)
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    mask_transformations = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
    ])

    train_dataset = VOCSegmentation('./data', image_set='train', transform=transformations, target_transform=mask_transformations)
    train_data = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = VOCSegmentation('./data', image_set='val', transform=transformations, target_transform=mask_transformations)
    test_data = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    loss_fn = CrossEntropyLoss(reduction='mean', ignore_index=255)


    train(
            optimizer=optimizer,
            model=model,
            fn_loss=loss_fn,
            train=train_data,
            test=test_data,
            scheduler=scheduler,
            device=device
            )




if __name__ == "__main__":
    main()