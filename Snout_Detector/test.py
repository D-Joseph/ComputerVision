import argparse
import torch
import time
from torch.utils.data import DataLoader
from model import SnoutNet
from SnoutDataset import SnoutDataset
import matplotlib.pyplot as plt


def test(model: SnoutNet, loader: DataLoader, device: torch.device, dataset: SnoutDataset, output_location: str):
    model.eval()
    start = time.time()
    min_distance, max_distance, sum_distance, sum_squared_distance, num_images = float('inf'), float('-inf'), 0, 0, len(dataset)
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device=device)
            lbls = lbls.to(device=device)

            predictions = model(imgs)

            # Calculate Euclidean distances between predicted and actual points
            distances = torch.sqrt((predictions[:, 0] - lbls[:, 0])**2 + (predictions[:, 1] - lbls[:, 1])**2)

            min_distance = min(min_distance, torch.min(distances).item()) 
            max_distance = max(max_distance, torch.max(distances).item())  
            sum_distance += torch.sum(distances).item() 
            sum_squared_distance += torch.sum(distances**2).item() 

            # Code for outputting a single image
            
            # img = imgs[0].permute(1, 2, 0).numpy() 
            # label = [lbls[0][0], lbls[0][1], predictions[0][0], predictions[0][1]]
            # print(label)
            # plt.imshow(img)
            # plt.scatter([label[0]], [label[1]], color='red', marker='x') 
            # plt.scatter([label[2]], [label[3]], color='yellow', marker='o') 
            # plt.savefig('test_output.png')

    # Calculate mean and standard deviation
    mean_distance = torch.tensor(sum_distance / num_images)
    std_dev = torch.sqrt((sum_squared_distance / num_images) - (mean_distance**2))

    end = time.time()

    print(f"Training Completed:\nTime Elapsed: {(end-start)/60:.2f} minutes\nMinimum Distance: {min_distance}\nMaximum Distance: {max_distance}\nAverage Distance: {mean_distance}\nStandard Deviation: {std_dev}")

def main():
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-d', metavar='model_data_directory', required=True, type=str, help='folder containing training information (i.e. weights); dir for outputs')
    argParser.add_argument('-i', metavar='images_directory', type=str, help='absolute path to images directory, defaults to ./oxford-iiit-pet-noses/images-original/images', default='./oxford-iiit-pet-noses/images-original/images')
    argParser.add_argument('-l', metavar='labels', type=str, help='absolute path to labels file, defaults to ./oxford-iiit-pet-noses/train_noses.txt', default='./oxford-iiit-pet-noses/train_noses.txt')
    argParser.add_argument('-b', metavar='batch_size', type=int, help='batch size, defaults to 64', default=64)
    args = argParser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    output_dir = args.d
    model = SnoutNet()
    model.load_state_dict(torch.load(f'{args.d}/weights.pth')) # Apply weights from training
    model.to(device=device)

    dataset = SnoutDataset(args.i, args.l)
    dataloader = DataLoader(
        dataset,
        batch_size=args.b,
        shuffle=True
    )

    test(model, dataloader, device, dataset, output_dir)

if __name__ == '__main__':
    main()