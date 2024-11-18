from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

def get_data_loaders(batch_size: int = 32, num_workers: int = 0):
    """ Get the CIFAR-100 data loaders. """
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    ])#yeah ik random ass values : https://gist.github.com/weiaicunzai/e623931921efefd4c331622c344d8151

    
    train = CIFAR100(root='./data', train=True, download=True, transform=transform)
    test = CIFAR100(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader