"""
Takes in an index from user input and displays the corresponding image in MNIST
"""

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST


train_transform = transforms.Compose([transforms.ToTensor()])
train_set = MNIST('./data/mnist', train=True, download=True, transform=train_transform)
idx = int(input("What image index do you want to view?: "))
plt.imshow(train_set.data[idx], cmap='gray')
plt.savefig(f'./outputs/MNIST - Index {idx}.png')