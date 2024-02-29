"""
Contains functionality to train and test a model.
"""

import torch
from accuracy import accuracy_fn

def train(model: torch.nn.Module,
          dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          accuracy_fn=accuracy_fn):
  train_loss, train_acc = 0, 0

  model.to(device)
  model.train()
  for X, y in dataloader:
    X, y = X.to(device), y.to(device)

    y_pred = model(X)

    loss = loss_fn(y_pred, y)
    train_loss += loss

    acc = accuracy_fn(y_true=y,
                      y_pred=y_pred.argmax(axis=1))
    train_acc += acc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss /= len(dataloader)
  train_acc /= len(dataloader)

  return train_loss, train_acc
  

def test(model: torch.nn.Module,
         dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         device: torch.device,
         accuracy_fn=accuracy_fn):
  test_loss, test_acc = 0, 0

  model.to(device)
  model.eval()
  with torch.inference_mode():
    for X, y in dataloader:
      X, y = X.to(device), y.to(device)

      y_pred = model(X)

      loss = loss_fn(y_pred, y)
      test_loss += loss

      acc = accuracy_fn(y_true=y,
                        y_pred=y_pred.argmax(axis=1))
      test_acc += acc
    
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)

  return test_loss, test_acc
