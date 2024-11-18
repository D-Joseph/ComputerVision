from torchvision.datasets import CIFAR100
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader

def get_data_loaders(batch_size: int = 32, num_workers: int = 2):
    """ Get the CIFAR-100 data loaders. """
    train_transform = Compose([
        RandomCrop(32, padding=4),  # Augmentation: Random cropping with padding
        RandomHorizontalFlip(),    # Augmentation: Horizontal flip
        Resize((224, 224)),        # Resize for AlexNet or other models
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 normalization
    ])
    
    test_transform = Compose([
        Resize((224, 224)), 
        ToTensor(),
        Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 normalization
    ])
    
    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=train_transform)
    test_dataset = CIFAR100(root="./data", train=False, download=True, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    
    return train_loader, test_loader