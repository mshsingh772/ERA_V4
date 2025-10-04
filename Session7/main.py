import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
from torchvision import transforms
from models.mynet import MyNet
from utils.param_utils import count_params
# from utils.plot_utils import plot_loss
from utils.plot_utils import plot_loss_accuracy
from train import train
from test import test
import config

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np
import os
from datetime import datetime

from models.mynet import MyNet
from utils.param_utils import count_params
from utils.plot_utils import plot_loss_accuracy
from utils.logger import Logger  # ✅ Custom logger
from train import train
from test import test
import config
def logprint(msg):
    print(msg)
    logger.log(msg)
mean, std = config.mean, config.std
# Transforms
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.3),
    A.CoarseDropout(
        num_holes_range=(1, 1),
        hole_height_range=(16, 16),
        hole_width_range=(16, 16),
        fill=tuple([int(m * 255) for m in mean]),
        p=0.5
    ),
    # A.CoarseDropout(max_holes=1, max_height=16, max_width=16,
    #                 min_holes=1, min_height=16, min_width=16,
    #                 fill_value=tuple([int(x*255) for x in config.mean]), p=0.5),
    A.Normalize(mean=config.mean, std=config.std),
    ToTensorV2()
])

test_transforms = A.Compose([
    A.Normalize(mean=config.mean, std=config.std),
    ToTensorV2()
])

# Albumentations Dataset Wrapper
class AlbumentationsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        image = np.array(image)
        image = self.transform(image=image)['image']
        return image, label
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Logger Setup ===
    log_file = f"logs/training_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    global logger
    logger = Logger(log_file)

    logprint("Training session started.")
    logprint(f"Using device: {device}")

    # === Dataset and Dataloader Setup ===
    train_dataset = CIFAR10(root='./data', train=True, download=True)
    test_dataset = CIFAR10(root='./data', train=False, download=True)

    train_ds = AlbumentationsDataset(train_dataset, train_transforms)
    test_ds = AlbumentationsDataset(test_dataset, test_transforms)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

    # === Model Setup ===
    model = MyNet().to(device)
    total_params = count_params(model)
    logprint(f"Model: MyNet")
    logprint(f"Total Trainable Parameters: {total_params}\n")

    optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    criterion = torch.nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    # === Training Loop ===
    for epoch in range(1, config.EPOCHS + 1):
        logprint(f"\nEpoch {epoch}/{config.EPOCHS}")

        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
        test_loss, test_acc = test(model, device, test_loader, criterion)

        scheduler.step()

        logprint(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logprint(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

    # === Save Final Plot ===
    os.makedirs("logs", exist_ok=True)
    plot_path = os.path.join("logs", "loss_accuracy_plot.png")
    plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies, save_path=plot_path)

    logprint("\nTraining complete!")
    logprint(f"Logs saved to: {log_file}")
    logprint(f"Loss and Accuracy plot saved to: {plot_path}")

    logger.close()

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     train_dataset = CIFAR10(root='./data', train=True, download=True)
#     test_dataset = CIFAR10(root='./data', train=False, download=True)

#     train_ds = AlbumentationsDataset(train_dataset, train_transforms)
#     test_ds = AlbumentationsDataset(test_dataset, test_transforms)

#     train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
#     test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)

#     model = MyNet().to(device)
#     print(f"Total Parameters: {count_params(model)}")

#     optimizer = torch.optim.SGD(model.parameters(), lr=config.LR, momentum=0.9, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
#     criterion = torch.nn.CrossEntropyLoss()

#     train_losses, test_losses = [], []

#     for epoch in range(1, config.EPOCHS + 1):
#         print(f"\nEpoch {epoch}/{config.EPOCHS}")
#         train_loss, train_acc = train(model, device, train_loader, optimizer, criterion)
#         test_loss, test_acc = test(model, device, test_loader, criterion)
#         scheduler.step()
#         print(f"Train Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
#         print(f"Test  Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%")
#         train_losses.append(train_loss)
#         test_losses.append(test_loss)

#     plot_loss(train_losses, test_losses)

if __name__ == "__main__":
    from torchsummary import summary
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyNet().to(device)
    # Print model summary
    summary(model, input_size=(3, 32, 32)) 
    main()
