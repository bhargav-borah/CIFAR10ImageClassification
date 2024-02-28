"""
Contains functionality to fit a model
to the given data and track the loss and accuracy for each epoch.
"""
import torch
from accuracy import accuracy_fn

def fit(model: torch.nn.Module,
        epochs: int,
        train_dataloader: torch.utils.data.DataLoader,
        test_dataloader: torch.utils.data.DataLoader,
        loss_fn: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device=device,
        accuracy_fn=accuracy_fn,
        patience: int=5,
        min_delta: float=0.001,
        wait: int=0):
  history = {
      'train_losses': [],
      'train_accuracies': [],
      'test_losses': [],
      'test_accuracies': [],
  }

  best_loss = np.Inf
  best_state_dict = model.state_dict()

  start_time = time()

  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train(model=model,
                                  dataloader=train_dataloader,
                                  loss_fn=loss_fn,
                                  optimizer=optimizer,
                                  device=device,
                                  accuracy_fn=accuracy_fn)
    history['train_losses'].append(train_loss.item())
    history['train_accuracies'].append(train_acc.item())

    test_loss, test_acc = test(model=model,
                               dataloader=test_dataloader,
                               loss_fn=loss_fn,
                               device=device,
                               accuracy_fn=accuracy_fn)
    history['test_losses'].append(test_loss.item())
    history['test_accuracies'].append(test_acc.item())

    print(f'Training loss:  {train_loss} | Training accuracy: {train_acc} | Test loss: {test_loss} | Test accuracy: {test_acc}')

    if test_loss < best_loss - min_delta:
      best_loss = test_loss
      best_state_dict = model.state_dict()
      wait = 0
    else:
      wait += 1
      if wait >= patience:
        print('Early stopping!')
        break

  end_time = time()
  history['training_time'] = end_time - start_time
  history['device'] = device
  history['best_state_dict'] = best_state_dict

  return history
