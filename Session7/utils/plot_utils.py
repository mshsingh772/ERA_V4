import matplotlib.pyplot as plt

def plot_loss_accuracy(train_losses, test_losses, train_accuracies, test_accuracies, save_path=None):
    epochs = range(1, len(train_losses) + 1)

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))

    # Plot Loss
    axs[0].plot(epochs, train_losses, label='Train Loss', color='blue')
    axs[0].plot(epochs, test_losses, label='Test Loss', color='orange')
    axs[0].set_title("Training and Validation Loss")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(epochs, train_accuracies, label='Train Accuracy', color='green')
    axs[1].plot(epochs, test_accuracies, label='Test Accuracy', color='red')
    axs[1].set_title("Training and Validation Accuracy")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()
