from typing import Optional

import torch
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(
      self,
      dataset_size: int = 100,
      channels: int = 3,
      image_width: int = 256,
      image_height: int = 256,
      n_classes: Optional[int] = None,
    ):
        self.dataset_size = dataset_size
        self.n_classes = n_classes

        self.data = torch.randn(dataset_size, channels, image_height, image_width)

        if self.n_classes:
            self.labels = torch.randint(low=0, high=self.n_classes, size=dataset_size)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index: int):
        if self.n_classes:
            return {'image': self.data[index], 'label': self.labels[index]}

        return {'image': self.data[index]}
