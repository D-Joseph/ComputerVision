
import argparse
import torch
import time
import os
from torch.utils.data import DataLoader
from model import get_model
from data import get_data_loaders
import matplotlib.pyplot as plt

def test(model, test_loader, device):


    # Test the model
    top1_correct, top5_correct = 0, 0
    total_samples = len(test_loader.dataset)
    start = time.time()

    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(test_loader):
            
            imgs = imgs.to(device=device)
            lbls = lbls.to(device=device)

            predictions = model(imgs)

            #top 1 accuracy
            _, top1_predictions = torch.max(predictions, 1)
            top1_correct += (top1_predictions == lbls).sum().item()

            #top 5 accuracy
            _, top5_predictions = torch.topk(predictions, 5, dim=1)
            top5_correct += torch.sum(lbls.unsqueeze(1) == top5_predictions).item()
            
    top1_error = 1 - (top1_correct/total_samples)
    top5_error = 1 - (top5_correct/total_samples)
    return top1_error, top5_error, time.time() - start

def save_error_rates(top1_error, top5_error, model_name, output_path):
    open(output_path, 'a').close()
    with open(output_path, "w") as f:
        f.write(f"Top 1 Error {model_name}: {top1_error}\n")
        f.write(f"Top 5 Error {model_name}: {top5_error}\n")

def main():
    parser = argparse.ArgumentParser(description='Test the model')
    parser.add_argument('--weights', type=str, default='outputs', help='The path to the model weights')
    parser.add_argument('--model', type=str, default='resnet18', help='The model to test')
    parser.add_argument('--device', type=str, default='cuda', help='The device to test on')
    parser.add_argument('--output_location', type=str, default='test_imgs', help='The location to save the output images')
    args = parser.parse_args()

    # Load the model
    model = get_model(args.model)
    device = torch.device(args.device)
    weights_path = os.path.join(args.weights, args.model, 'model_convergence.pth')
    output_path = os.path.join(args.weights,args.model, f"{args.model}_error_rates.txt")
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    test_loader = get_data_loaders()[1]
    top1_error, top5_error, total_time = test(model, test_loader, device)
    save_error_rates(top1_error, top5_error, args.model, output_path)


if __name__ == '__main__':
    main()