"""
Contains functionality to create PyTorch DataLoaders
from the CIFAR-10 dataset
"""
from pathlib import Path
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os

def get_cifar10_dataloaders(data_dir: str,
                            transform: transforms.Compose,
                            batch_size: int=32,
                            num_workers: int=os.cpu_count()):
  """
  Prepare DataLoader objects for the CIFAR-10 dataset

  Args:
  - data_dir (str): The directory where the CIFAR-10 dataset will be stored.
  - transform (torchvision.transforms.transforms.Compose): A composition of image transformations to apply to the dataset.
  - batch_size (int, optional): Number of samples per batch to load (default is 32).
  - num_workers (int, optional): Number of subprocesses to use for data loading (default is the number of CPU cores).

  Returns:
  - train_dataloader (torch.utils.data.DataLoader): DataLoader object for the training data.
  - test_dataloader (torch.utils.data.DataLoader): DataLoader object for the test data.
  - class_names (list): A list of class names present in the CIFAR-10 dataset.

  The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class.
    This function loads the CIFAR-10 dataset from the specified `data_dir`, applies the specified `transformations`,
    and creates DataLoader objects for both the training and test datasets. The training DataLoader shuffles the data,
    while the test DataLoader does not shuffle the data.

  Example:
  ```python
  import torchvision.transforms as transforms

  # Define transformations
  transform = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # normalize to [-1, 1]
  ])

  # Get dataloaders
  train_dataloader, test_dataloader, class_names = get_cifar10_dataloaders(
      data_dir='/path/to/your/CIFAR-10/dataset',
      transform=transform,
      batch_size=32,
      num_workers=4
  )
  ```
  """
  # Set up the path for root directory
  data_dir = Path(data_dir)

  # Get the training data
  train_data = CIFAR10(root=data_dir,
                       train=True,
                       transform=transform,
                       target_transform=None,
                       download=True)

  # Get the test data
  test_data = CIFAR10(root=data_dir,
                      train=False,
                      transform=transform,
                      target_transform=None,
                      download=True)

  # Get the class names in a list
  class_names = train_data.classes

  # Prepare the training dataloader
  train_dataloader = DataLoader(dataset=train_data,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=num_workers)

  # Prepare the test dataloader
  test_dataloader = DataLoader(dataset=test_data,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=num_workers)

  return train_dataloader, test_dataloader, class_names
