import torch
from src.dataset import get_data_loaders
from src.model import MNISTModel
from src.trainer import Trainer
from src.utils import plot_prediction

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=64)
    
    # Initialize model
    model = MNISTModel()
    
    # Print model parameters
    num_params = count_parameters(model)
    print(f"\nModel Parameters: {num_params:,}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        learning_rate=0.001,
        device=device
    )
    
    # Train the model
    trainer.train(n_epochs=18)
    
    # Plot some predictions
    # plot_prediction(model, test_loader, device)

if __name__ == "__main__":
    main() 

