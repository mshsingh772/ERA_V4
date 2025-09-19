import torch
import matplotlib.pyplot as plt

def plot_prediction(model, data_loader, device, num_images=5):
    """Plot some predictions from the model."""
    model.eval()
    images, labels = next(iter(data_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
    
    fig = plt.figure(figsize=(12, 4))
    for idx in range(num_images):
        ax = fig.add_subplot(1, num_images, idx+1)
        ax.imshow(images[idx].cpu().squeeze(), cmap='gray')
        ax.set_title(f'Pred: {predictions[idx]}\nTrue: {labels[idx]}')
        ax.axis('off')
    plt.tight_layout()
    plt.show() 