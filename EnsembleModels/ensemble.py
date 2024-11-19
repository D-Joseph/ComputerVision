import torch
import torch.nn as nn
import torch.nn.functional as F
from model import get_model
from data import get_data_loaders
import argparse
import os

def save_accuracy(accuracy, method, ensemble_type, output_path, description=""):
    top1_error = 1 - accuracy
    print(f"Top1 error rate for {description} and {ensemble_type} ensemble using {method} method: {top1_error:.4f}")
    with open(output_path, "a") as f:
        f.write(f"Top1 error rate for {description} and {ensemble_type} ensemble using {method} method: {top1_error:.4f}\n")
    print(f"Finished {description}. Results logged to {output_path}")

def ensemble_probability_averaging(models, test_loader, device, description):
    correct = 0 

    total = len(test_loader.dataset)

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device=device)
            lbls = lbls.to(device=device)

            avg_probs = torch.zeros(lbls.size(0), 100).to(device=device) #probabilities tensor for each model

            for model in models:
                model.eval()
                outputs = model(imgs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                avg_probs += probs

            
            avg_probs /= len(models) #average the probabilities
            predictions = torch.argmax(avg_probs, dim=1)  #find the max prob among each row
            correct += (predictions == lbls).sum().item() #count the correct predictions

    accuracy = correct / total
    save_accuracy(accuracy, "probability_averaging", "ensemble", "outputs/ensemble_accuracy.txt", description)

def ensemble_majority_voting(models, test_loader, device, description):
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # Initialize a vote count tensor for each class
            votes = torch.zeros(lbls.size(0), 100, device=device)  # 100 = number of classes

            for model in models:
                model.eval()
                outputs = model(imgs)  # Get model predictions
                predictions = torch.argmax(outputs, dim=1)  # Find the class index with max value
                
                # Update votes for each class using one-hot encoding
                votes += F.one_hot(predictions, num_classes=100).float()

            # Determine the majority vote for each image
            majority_vote = torch.argmax(votes, dim=1)

            # Count correct predictions
            correct += (majority_vote == lbls).sum().item()
    
    # Calculate accuracy
    accuracy = correct / total

    # Save accuracy to a file
    save_accuracy(accuracy, "majority_voting", "ensemble", "outputs/ensemble_accuracy.txt",description)

def ensemble_max_probability(models, test_loader, device, description):
    correct = 0
    total = len(test_loader.dataset)

    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs = imgs.to(device)
            lbls = lbls.to(device)

            # Initialize a tensor to store the maximum probabilities
            max_probs = torch.zeros(lbls.size(0), 100, device=device)

            for model in models:
                model.eval()
                outputs = model(imgs)  # Get model predictions
                probs = F.softmax(outputs, dim=1)  # Compute class probabilities
                max_probs = torch.max(max_probs, probs)  # Update max probabilities

            # Determine the class with the maximum probability
            predictions = torch.argmax(max_probs, dim=1)

            # Count correct predictions
            correct += (predictions == lbls).sum().item()

    # Calculate accuracy
    accuracy = correct / total

    # Save accuracy to a file
    save_accuracy(accuracy, "max_probability", "ensemble", "outputs/ensemble_accuracy.txt", description )

def run_ensembles(model1_weights, model2_weights, model3_weights, test_loader, device, description):
    # Load trained models with specified weights
    model1 = get_model("resnet18", 100)
    model1.load_state_dict(torch.load(model1_weights, map_location=device))
    model1.to(device)

    model2 = get_model("alexnet", 100)
    model2.load_state_dict(torch.load(model2_weights, map_location=device))
    model2.to(device)

    model3 = get_model("vgg16", 100)
    model3.load_state_dict(torch.load(model3_weights, map_location=device))
    model3.to(device)

    # Perform ensemble methods
    print(f"Running ensembles for {description}...")
    models = [model1, model2, model3]
    ensemble_max_probability(models, test_loader, device, description)
    ensemble_probability_averaging(models, test_loader, device, description)
    ensemble_majority_voting(models, test_loader, device, description)

def main():
    parser = argparse.ArgumentParser(description="Test the ensemble models")
    parser.add_argument('--data', type=str, default='data', help='The path to the data directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for test loader')
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers for test loader')
    parser.add_argument('--weights', type=str, default='outputs', help='The path to the model weights')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    data_dir = args.data
    batch_size = args.batch_size
    num_workers = args.num_workers

    # Paths to specific weight files
    epoch_5_weights = {
        "resnet18": f'{args.weights}/resnet18/model_epoch_5.pth',
        "alexnet": f'{args.weights}/alexnet/model_epoch_5.pth',
        "vgg16": f'{args.weights}/vgg16/model_epoch_5.pth'
    }

    full_converge_weights = {
        "resnet18": f'{args.weights}/resnet18/model_convergence.pth',
        "alexnet": f'{args.weights}/alexnet/model_convergence.pth',
        "vgg16":f'{args.weights}/vgg16/model_convergence.pth'
    }

    # Load test data
    test_loader = get_data_loaders(batch_size, num_workers, data_dir)[1]

    # Run ensembles for weights at epoch 5
    run_ensembles(epoch_5_weights["resnet18"], epoch_5_weights["alexnet"], epoch_5_weights["vgg16"], 
                  test_loader, device, description="Epoch 5 Weights")

    # Run ensembles for fully converged weights
    run_ensembles(full_converge_weights["resnet18"], full_converge_weights["alexnet"], full_converge_weights["vgg16"], 
                  test_loader, device, description="Fully Converged Weights")


if __name__ == "__main__":
    main()



