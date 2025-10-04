import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    for data, target in tqdm(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
    return running_loss / len(train_loader), 100. * correct / total
