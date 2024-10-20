from SnoutDataset import SnoutDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


train_dataloader = DataLoader(
    SnoutDataset('./oxford-iiit-pet-noses/images-original/images', './oxford-iiit-pet-noses/train_noses.txt'),
    batch_size=1, 
    shuffle=True
)

train_features, train_labels = next(iter(train_dataloader))
img = train_features[0].permute(1, 2, 0).numpy() 
label = [train_labels[0][0], train_labels[1][0]]
plt.imshow(img)
plt.scatter([label[0]], [label[1]], color='red', marker='x') 
plt.savefig('resized_image.png')
