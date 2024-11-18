import argparse
import torch
import time
import os
from torch.utils.data import DataLoader
from model import get_model
from data import get_data_loaders


def test(model, test_loader, device):
    # Test the model
    top1_correct, top5_correct = 0, 0
    total_samples = len(test_loader.dataset)
    start = time.time()

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device=device)
            lbls = lbls.to(device=device)

            predictions = model(imgs)

            # Top-1 accuracy
            _, top1_predictions = torch.max(predictions, 1)
            top1_correct += (top1_predictions == lbls).sum().item()

            # Top-5 accuracy
            _, top5_predictions = torch.topk(predictions, 5, dim=1)
            top5_correct += torch.sum(lbls.unsqueeze(1) == top5_predictions).item()

    top1_error = 1 - (top1_correct / total_samples)
    top5_error = 1 - (top5_correct / total_samples)
    return top1_error, top5_error, time.time() - start


def save_error_rates(top1_error, top5_error, model_name, epoch_label, output_path):
    with open(output_path, "a") as f:
        f.write(f"Error Rates for {model_name} ({epoch_label}):\n")
        f.write(f"  Top 1 Error: {top1_error:.4f}\n")
        f.write(f"  Top 5 Error: {top5_error:.4f}\n\n")


def main():
    parser = argparse.ArgumentParser(description="Test the model")
    parser.add_argument('--data', type=str, default='data', help='The path to the data directory')
    parser.add_argument('--weights', type=str, default='outputs', help='The path to the model weights')
    parser.add_argument('--model', type=str, default='resnet18', help='The model to test')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for test loader')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for test loader')

    args = parser.parse_args()

    # Load the model
    model = get_model(args.model, 100)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Paths to the weight files
    convergence_weights_path = os.path.join(args.weights, args.model, 'model_convergence.pth')
    epoch5_weights_path = os.path.join(args.weights, args.model, 'model_epoch_5.pth')

    # Output file for error rates
    output_path = os.path.join(args.weights, args.model, f"{args.model}_error_rates.txt")

    # Get the data loaders
    test_loader = get_data_loaders(batch_size, num_workers, data_dir)[1]

    # Test for weights at epoch 5
    model.load_state_dict(torch.load(epoch5_weights_path, map_location=device))
    model.to(device)
    top1_error, top5_error, total_time = test(model, test_loader, device)
    save_error_rates(top1_error, top5_error, args.model, "Epoch 5", output_path)

    # Test for weights at convergence
    model.load_state_dict(torch.load(convergence_weights_path, map_location=device))
    model.to(device)
    top1_error, top5_error, total_time = test(model, test_loader, device)
    save_error_rates(top1_error, top5_error, args.model, "Convergence", output_path)


if __name__ == "__main__":
    main()
