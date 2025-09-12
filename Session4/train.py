import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mnist_model import LightMNIST, train_model, test_model, count_parameters

def main():
    device = torch.device("cpu")
    
    # Training transform with minimal augmentation
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomAffine(
            degrees=1,              # Very small rotation
            translate=(0.02, 0.02), # Small translation
            scale=(0.98, 1.02)    # Minimal scaling
        ),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=train_transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=500)
    
    model = LightMNIST().to(device)
    
    # Simple SGD with momentum
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.02,
        momentum=0.9,
        nesterov=True
    )
    
    n_params = count_parameters(model)
    print(f"Number of parameters: {n_params}")
    
    train_acc = train_model(model, device, train_loader, optimizer, None, epoch=1)
    test_acc = test_model(model, device, test_loader)
    
    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    
    torch.save(model.state_dict(), "mnist_model.pth")
    
    return train_acc, n_params

if __name__ == "__main__":
    main() 