import os
from typing import Callable, Dict, Union

import numpy as np
import torch

from typing import List, Union
import h5py
import numpy as np
from omegaconf import DictConfig
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset
import socket


class So2Sat(Dataset):
    """So2Sat"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None

    def __init__(
        self, path: str, transform, channels: List[int], split: str  ## split: either train, valid, or test
    ) -> None:
        """Initialize the dataset."""
        super().__init__()

        self.channels = torch.tensor([c for c in channels])

        self.transform = transform
        ## read h5py file from `path`
        if split == "train":
            path = os.path.join(path, "training.h5")
        else:  ## we use the same validation set for both validation and test.
            ## for validation, we use some channels.
            ## for test, we use all channels.
            path = os.path.join(path, "validation.h5")
        self.file = h5py.File(path, "r")
        self.path = path

    def __getitem__(self, index):
        ## https://github.com/zhu-xlab/So2Sat-LCZ42
        ## use both sen1 and sen2
        img_chw = np.concatenate(
            [
                self.file["sen1"][index].astype("float32"),
                self.file["sen2"][index].astype("float32"),
            ],
            axis=-1,
        )
        ## reorder the channels to c, h, w
        img_chw = np.transpose(img_chw, (2, 0, 1))

        label = self.file["label"][index].astype(int)
        img_chw = self.transform(img_chw)

        channels = self.channels.numpy()
        img_chw = img_chw[channels]

        if sum(label) > 1:
            raise ValueError("More than one positive")

        for i, y in enumerate(label):
            if y == 1:
                label = i
                break
        out = {"image": torch.tensor(img_chw.copy()).float(), "channels": channels, "label": label}
        return out

    def __len__(self) -> int:
        return len(self.file["label"])

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)
