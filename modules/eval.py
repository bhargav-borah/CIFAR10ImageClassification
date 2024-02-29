"""
Contains functionality to evaluate a trained model.
"""
import torch
from accuracy import accuracy_fn

def eval_model(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device,
               accuracy_fn=accuracy_fn):
  """
  Evaluate the performance of a PyTorch model on a given dataset.

  Args:
  - model (torch.nn.Module): The PyTorch model to evaluate.
  - dataloader (torch.utils.data.DataLoader): DataLoader providing batches of data.
  - loss_fn (torch.nn.Module): The loss function used for evaluation.
  - device (torch.device): The device to perform the evaluation (e.g., 'cuda' or 'cpu').
  - accuracy_fn (function): The function to calculate the accuracy score. Default is 'accuracy_fn'.

  Returns:
  - Tuple[float, float]: A tuple containing the average loss and average accuracy.
  """
  avg_loss, avg_acc = 0, 0

  model.to(device)
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)

      y_pred = model(X)

      loss = loss_fn(y_pred, y)
      avg_loss += loss

      acc = accuracy_fn(y_true=y,
                        y_pred=y_pred.argmax(axis=1))
      avg_acc += acc

    avg_loss /= len(dataloader)
    avg_acc /= len(dataloader)

  return avg_loss.item(), avg_acc.item()
