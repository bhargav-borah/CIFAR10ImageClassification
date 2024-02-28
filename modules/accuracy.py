"""
Contains functionality to calculate the accuracy score.
"""
import torch

def accuracy_fn(y_true, y_pred):
  correct = torch.sum(torch.eq(y_true, y_pred))
  total = len(y_true)

  return correct / total
