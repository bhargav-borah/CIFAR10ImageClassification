"""
Contains functionality to plot training and test accuracies and losses.
"""
import matplotlib.pyplot as plt

def plot_loss_curves(train_losses, test_losses):
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

  axs[0].plot(range(len(train_losses)), train_losses, color='blue')
  axs[0].set_title('Training Loss Curve')
  axs[0].set_xlabel('Number of Epochs')
  axs[0].set_ylabel('Loss')

  axs[1].plot(range(len(test_losses)), test_losses, color='orange')
  axs[1].set_title('Test Loss Curve')
  axs[1].set_xlabel('Number of Epochs')
  axs[1].set_ylabel('Loss')

  plt.show()

def plot_accuracy_curves(train_accuracies, test_accuracies):
  fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))

  axs[0].plot(range(len(train_accuracies)), train_accuracies)
  axs[0].set_title('Training Accuracy Curve')
  axs[0].set_xlabel('Number of Epochs')
  axs[0].set_ylabel('Accuracy')

  axs[1].plot(range(len(test_accuracies)), test_accuracies)
  axs[1].set_title('Test Accuracy Curve')
  axs[1].set_xlabel('Number of Epochs')
  axs[1].set_ylabel('Accuracy')

  plt.show()
