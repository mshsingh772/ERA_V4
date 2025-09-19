import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

class Trainer:
    def __init__(self, model, train_loader, test_loader, 
                 learning_rate=0.001, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(self.test_loader, desc="Testing "):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, n_epochs=5, model_save_path="models/mnist_model.pth"):
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        
        best_accuracy = 0.0
        print("\nStarting training...")
        print("-" * 60)
        
        for epoch in range(n_epochs):
            print(f"\nEpoch [{epoch+1}/{n_epochs}]")
            
            # Training phase
            train_loss, train_acc = self.train_epoch()
            
            # Testing phase
            test_loss, test_acc = self.evaluate()
            
            # Print metrics
            print("\nMetrics:")
            print(f"{'Train Loss':>12}: {train_loss:.4f}  {'Train Accuracy':>16}: {train_acc:.2f}%")
            print(f"{'Test Loss':>12}: {test_loss:.4f}  {'Test Accuracy':>16}: {test_acc:.2f}%")
            
            # Save best model
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                torch.save(self.model.state_dict(), model_save_path)
                print(f"\n🔥 New best model saved! (Accuracy: {test_acc:.2f}%)")
            
            print("-" * 60)
        
        print(f"\nTraining completed! Best test accuracy: {best_accuracy:.2f}%")