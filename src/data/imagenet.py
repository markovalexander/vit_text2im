from pathlib import Path
from typing import Any, Callable, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageNet


class ImageNetBase(ImageNet):
    def __init__(self, root: str, split: str,
                 transform: Optional[Callable] = None) -> None:
        super().__init__(root=root, split=split, transform=transform)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample, target = super().__getitem__(index)

        return {'image': sample, 'class': torch.tensor([target])}


class ImageNetTrain(ImageNetBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,
                 resize_ratio: float = 0.75) -> None:

        transform = T.Compose([
            T.Resize(resolution),
            T.RandomCrop(resolution),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        super().__init__(root=root, split='train', transform=transform)


class ImageNetValidation(ImageNetBase):
    def __init__(self, root: str,
                 resolution: Union[Tuple[int, int], int] = 256,) -> None:

        if isinstance(resolution, int):
            resolution = (resolution, resolution)

        transform = T.Compose([
            T.Resize(resolution),
            T.CenterCrop(resolution),
            T.ToTensor()
        ])

        super().__init__(root=root, split='val', transform=transform)


def get_loaders(
  root_path: Path,
  batch_size: int,
  num_workers: int,
  return_train: bool = True,
  return_test: bool = True,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:

    assert return_train or return_test, "should return at least something"

    if return_train:
        train_dataset = ImageNetTrain(root_path.as_posix())
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)

    if return_test:
        val_dataset = ImageNetValidation(root_path.as_posix())
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=num_workers)

    if return_train and return_test:
        return train_loader, val_loader
    elif return_train:
        return train_loader
    elif return_test:
        return val_loader
