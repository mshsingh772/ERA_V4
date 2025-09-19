import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=64, data_dir="data"):
    """Create and return train and test data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    # Download and load test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"\nDataset sizes:")
    print(f"Training set: {len(train_dataset):,} images")
    print(f"Test set:     {len(test_dataset):,} images")
    print(f"Batch size:   {batch_size}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches:     {len(test_loader)}\n")

    return train_loader, test_loader 