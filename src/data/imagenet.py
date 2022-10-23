from pathlib import Path
from typing import Tuple, Union

from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

train_transforms = T.Compose([
    T.Resize(256),
    T.RandomCrop(256),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])
test_transforms = T.Compose([
    T.Resize(256),
    T.CenterCrop(256),
    T.ToTensor()
])

def get_loaders(
  root_path: Path,
  batch_size: int,
  num_workers: int,
  return_train: bool = True,
  return_test: bool = True,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:

    assert return_train or return_test, "should return at least something"

    if return_train:
        train_dataset = ImageFolder((root_path / 'train').as_posix(), transform=train_transforms)
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    if return_test:
        val_dataset = ImageFolder((root_path / 'val').as_posix(), transform=test_transforms)
        val_loader = DataLoader(val_dataset, 4, shuffle=False, num_workers=num_workers)

    if return_train and return_test:
        return train_loader, val_loader
    elif return_train:
        return train_loader
    elif return_test:
        return val_loader
