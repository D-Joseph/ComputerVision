from torchvision.models import resnet18
import torch
from torchvision.datasets import VOCSegmentation

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = resnet18(pretrained=True)
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
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
if __name__ == "__main__":
    main()