from pathlib import Path
from typing import Tuple, Union

import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10, CIFAR100

CIFARDataset = Union[CIFAR10, CIFAR100]

class CIFARDatasetBuilder:
    dataset_classes = {
        'cifar10': CIFAR10,
        'cifar100': CIFAR100,
    }

    train_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])

    def __init__(self, dataset_name: str):
        if dataset_name not in self.dataset_classes:
            raise ValueError('dataset is not supported yet!')

        self.dataset = self.dataset_classes[dataset_name]

    def get_train(self, path: Path, download: bool = False) -> CIFARDataset:
        if not download and not path.exists():
            raise ValueError('Given path does not exist')
        return self.dataset(root=path, transform=self.train_transforms, download=download)

    def get_test(self, path: Path, download: bool = False) -> CIFARDataset:
        return self.dataset(root=path, download=download, transform=T.ToTensor())


def get_cifar_loaders(
  name: str,
  batch_size: int,
  path: Path,
  download: bool = False,
  n_workers:int = 4,
) -> Tuple[DataLoader, DataLoader]:
    dataset_builder = CIFARDatasetBuilder(name)

    train_data = dataset_builder.get_train(path, download)
    test_data = dataset_builder.get_test(path, download)

    train_loader = DataLoader(train_data, batch_size, shuffle=True, num_workers=n_workers)
    test_loader = DataLoader(test_data, 4, shuffle=False)
    return train_loader, test_loader
