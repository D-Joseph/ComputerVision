import torch
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import time
import argparse
from torchvision.datasets import VOCSegmentation
from torch.nn import CrossEntropyLoss
from torchvision.transforms import InterpolationMode
from torchvision.models.segmentation import fcn_resnet50
from model import ResNet18Segmentation 
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt

def train( **kwargs) -> None:
    """
    Train the model.

    Args:
        epochs (Optional[int]): The number of epochs for training. Default is 30.
        kwargs: 
            - student: The student model to be trained.
            - teacher: The teacher model to be trained.
            - train: The data loader for training data.
            - test: The data loader for training data.
            - device: The device to train on ('cpu' or 'cuda').
            - optimizer: The optimizer to use for training.
            - scheduler: The learning rate scheduler.
    """
    print("Training Parameters")
    for kwarg in kwargs:
        print(kwarg, "=", kwargs[kwarg])
    student = kwargs['student']
    teacher = kwargs['teacher']
    epochs = kwargs['epochs']
    loss_fn = kwargs['loss_fn']
    teacher.eval()
    student.train()

    losses_train = []
    losses_val = []
    start = time.time()
    if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
        output_dir = './output/none'
    elif loss_fn == response_distillation:
        output_dir = './output/response'
    elif loss_fn == feature_distillation:
        output_dir = './output/feature'
    os.makedirs(f"{output_dir}", exist_ok=True)
    for epoch in range(1, epochs+1):
        student.train()
        print("Epoch:", epoch)
        loss_train = 0.0
        for data in kwargs['train']:
            imgs, lbls = data
            imgs = imgs.to(device=kwargs['device'])
            lbls = lbls.to(device=kwargs['device'])
            s_logits = student(imgs)
            t_logits = teacher(imgs)["out"]
            if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                loss = loss_fn(s_logits, lbls)
            else:
                loss = loss_fn(s_logits, t_logits, lbls, alpha=0.5, temperature=2)
            kwargs['optimizer'].zero_grad()
            loss.backward()
            kwargs['optimizer'].step()
            loss_train += loss.item()

        
        losses_train.append(loss_train/len(kwargs['train']))

        
        student.eval()  # Set the student model to evaluation mode
        loss_val = 0.0
        teacher_acc = 0.0
        student_acc = 0.0
        with torch.no_grad():
            for imgs, lbls in kwargs['test']:
                imgs = imgs.to(device=kwargs['device'])
                lbls = lbls.to(device=kwargs['device'])

                s_logits = student(imgs)
                t_logits = teacher(imgs)["out"]
                if isinstance(loss_fn, torch.nn.CrossEntropyLoss):
                    loss = loss_fn(s_logits, lbls)
                else:
                    loss = loss_fn(s_logits, t_logits, lbls, alpha=0.5, temperature=2)

                loss_val += loss.item()
                teacher_acc += calculate_accuracy(t_logits, lbls)
                student_acc += calculate_accuracy(s_logits, lbls)
        teacher_acc /= len(kwargs['test'])
        student_acc /= len(kwargs['test'])
        kwargs['scheduler'].step(loss_val/len(kwargs['test']))
        losses_val.append(loss_val / len(kwargs['test']))
        
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()), "Epoch:", epoch, "Training loss:", loss_train/len(kwargs['train']), "Validation loss:", loss_val / len(kwargs['test']))
        print("Teacher Accuracy:", teacher_acc, "Student Accuracy:", student_acc)

        filename = f"{output_dir}/weights.pth"
        print("Saving Weights to", filename)
        torch.save(student.state_dict(), filename)

        plt.figure(2, figsize=(12, 7))
        plt.clf()
        plt.plot(losses_train, label='train')
        plt.plot(losses_val, label='val')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=1)
        filename = f"{output_dir}/loss_plot.png"
        print('Saving to Loss Plot at', filename)
        plt.savefig(filename)
        # Ensure output directory exists
        
        accuracy_file = os.path.join(f"{output_dir}", "accuracy.txt")

        #Append the accuracies to the file
        with open(accuracy_file, "a") as f:
            f.write(f"Epoch {epoch}: Teacher Accuracy: {teacher_acc:.2f}%, Student Accuracy: {student_acc:.2f}%\n")


    end = time.time()
    elapsed_time = (end - start) / 60
    print("Training completed in", round(elapsed_time, 2), "minutes")
    #time_filename = './outputs/training_time.txt'

#TODO: Finish creating this new loss function
def response_distillation(s_logits, t_logits, label, alpha = 0.5, temperature = 2):
    ce_loss = torch.nn.CrossEntropyLoss()(s_logits, label)
    t_soft = torch.nn.functional.softmax(t_logits / temperature, dim=1)
    s_soft = torch.nn.functional.log_softmax(s_logits / temperature, dim=1)

    distillation_loss = torch.nn.functional.kl_div(s_soft, t_soft, reduction='batchmean') * (temperature ** 2)

    beta = 1 - alpha

    return alpha * ce_loss + beta * distillation_loss

def feature_distillation(s_logits, t_logits, label, alpha = 0.5, temperature = 2):
    ce_loss = torch.nn.CrossEntropyLoss()(s_logits, label)
    t_features = t_logits['out']
    s_features = s_logits
    if s_features.shape != t_features.shape:
        t_features = torch.nn.functional.interpolate(
        t_features, size=s_features.shape[2:], mode='bilinear', align_corners=False
    )

    distillation_loss = torch.nn.functional.mse_loss(s_features, t_features)

    beta = 1 - alpha

    return alpha * ce_loss + beta * distillation_loss

def calculate_accuracy(logits, labels):
        preds = torch.argmax(logits, dim=1)  # Get predicted classes (N, H, W)
        correct = (preds == labels).sum().item()  # Count correctly predicted pixels
        total = torch.numel(labels)  # Total number of pixels
        return 100 * correct / total


def main():
    parser = argparse.ArgumentParser(description="Train a model")
    
    parser.add_argument('-e', metavar='epochs', type=int, help='Number of epochs', default=30)
    parser.add_argument('-b', metavar='batch_size', type=int, help='Batch size', default=32)
    parser.add_argument(
        '--loss', 
        choices=['response', 'feature', 'none'], 
        help='Distillation method', 
        default='response'
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    student = ResNet18Segmentation(num_classes=21)
    teacher = fcn_resnet50(pretrained=True)
    student = student.to(device)
    teacher = teacher.to(device)

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    mask_transformations = transforms.Compose([
        transforms.Resize((256, 256), interpolation=InterpolationMode.NEAREST),
        transforms.Lambda(lambda x: torch.tensor(np.array(x, dtype=np.int64))),
    ])

    train_dataset = VOCSegmentation('./data', image_set='train', transform=transformations, target_transform=mask_transformations)
    train_data = DataLoader(train_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True)
    test_dataset = VOCSegmentation('./data', image_set='val', transform=transformations, target_transform=mask_transformations)
    test_data = DataLoader(test_dataset, batch_size=args.b, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(student.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    if args.loss == 'response':
        loss_fn = response_distillation
    elif args.loss == 'feature':
        loss_fn = feature_distillation
    else:
        loss_fn = torch.nn.CrossEntropyLoss()


    train(
            optimizer=optimizer,
            student=student,
            teacher=teacher,
            train=train_data,
            test=test_data,
            scheduler=scheduler,
            device=device,
            loss_fn=loss_fn,
            epochs=args.e
            )




if __name__ == "__main__":
    main()