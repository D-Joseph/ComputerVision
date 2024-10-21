import argparse
import torch
import time
from torch.utils.data import DataLoader
from model import SnoutNet
from SnoutDataset import SnoutDataset


def test(model: SnoutNet, loader: DataLoader, device: torch.device, output_location: str):
    start = time.time()

    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device=device)
            lbls = lbls.to(device=device)
            guess = model(imgs)

            # TODO: Take guess and calculate distance stats
    
    end = time.time()

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', metavar='model_data_directory', required=True, type=str, help='folder containing training information (i.e. weights); dir for outputs', default='./outputs_none')
    argParser.add_argument('-i', metavar='images_directory', type=str, help='absolute path to images directory, defaults to ./oxford-iiit-pet-noses/images-original/images', default='./oxford-iiit-pet-noses/images-original/images')
    argParser.add_argument('-l', metavar='labels', type=str, help='absolute path to labels file, defaults to ./oxford-iiit-pet-noses/train_noses.txt', default='./oxford-iiit-pet-noses/train_noses.txt')
    argParser.add_argument('-b', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    args = argParser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = args.d
    model = SnoutNet()
    model.load_state_dict(torch.load(f'{args.d}/weights.pth')) # Apply weights from training
    model.to(device=device)
    model.eval()

    dataloader = DataLoader(
        SnoutDataset(args.i, args.l),
        batch_size=args.b,
        shuffle=True
    )

    test(model, dataloader, device, output_dir)

    



if __name__ == '__main__':
    main()