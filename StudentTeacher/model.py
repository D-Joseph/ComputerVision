import torch
import torch.nn as nn
from torchvision import models

class ResNet18Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18Segmentation, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        
        self.encoder = nn.Sequential(
            resnet18.conv1,
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool,
            resnet18.layer1,
            resnet18.layer2,
            resnet18.layer3,
            resnet18.layer4,
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        )

        # Calculate and store the number of trainable parameters
        self.num_trainable_params = self.calculate_trainable_parameters()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        return x

    def calculate_trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# Access the number of trainable parameters
# print(f'Total number of trainable parameters: {model.num_trainable_params}')