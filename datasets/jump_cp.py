from typing import List, Union

import numpy as np
import pandas as pd
import torch
import os
from omegaconf import DictConfig, ListConfig
from tqdm import tqdm
from torch.utils.data.dataloader import default_collate

from torch.utils.data import Dataset
## get cur machine name
import socket
cur_machine = socket.gethostname()


def load_meta_data():
    PLATE_TO_ID = {"BR00116991": 0, "BR00116993": 1, "BR00117000": 2}
    FIELD_TO_ID = dict(zip([str(i) for i in range(1, 10)], range(9)))
    WELL_TO_ID = {}
    for i in range(16):
        for j in range(1, 25):
            well_loc = f"{chr(ord('A') + i)}{j:02d}"
            WELL_TO_ID[well_loc] = len(WELL_TO_ID)

    WELL_TO_LBL = {}
    # map the well location to the perturbation label
    base_path = "/projectnb/ivc-ml/chaupham/ChannelViT/data/jumpcp/platemap_and_metadata"
    if cur_machine == 'goat.bu.edu':
        base_path = "/scratch/chaupham/jumpcp/platemap_and_metadata"
    # "s3://insitro-research-2023-context-vit/jumpcp/platemap_and_metadata"

    PLATE_MAP = {
        "compound": f"{base_path}/JUMP-Target-1_compound_platemap.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_platemap.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_platemap.tsv",
    }
    META_DATA = {
        "compound": f"{base_path}/JUMP-Target-1_compound_metadata.tsv",
        "crispr": f"{base_path}/JUMP-Target-1_crispr_metadata.tsv",
        "orf": f"{base_path}/JUMP-Target-1_orf_metadata.tsv",
    }

    for perturbation in PLATE_MAP.keys():
        df_platemap = pd.read_parquet(PLATE_MAP[perturbation])
        df_metadata = pd.read_parquet(META_DATA[perturbation])
        df = df_metadata.merge(df_platemap, how="inner", on="broad_sample")

        if perturbation == "compound":
            target_name = "target"
        else:
            target_name = "gene"

        codes, uniques = pd.factorize(df[target_name])
        codes += 1  # set none (neg control) to id 0
        assert min(codes) == 0
        print(f"...{target_name} has {len(uniques)} unique values")
        WELL_TO_LBL[perturbation] = dict(zip(df["well_position"], codes))

    return PLATE_TO_ID, FIELD_TO_ID, WELL_TO_ID, WELL_TO_LBL


class JUMPCP(Dataset):
    """JUMPCP dataset"""

    normalize_mean: Union[List[float], None] = None
    normalize_std: Union[List[float], None] = None
    NUM_TOTAL_CHANNELS = 8

    def __init__(
        self,
        path: str,
        split: str,  # train, valid or test
        transform,
        channels: Union[List[int], None],
        channel_mask: bool = False,
        scale: float = 1,
        perturbation_list: ListConfig[str] = ["compound"],
        cyto_mask_path_list: ListConfig[str] = None
    ) -> None:
        """Initialize the dataset."""
        self.root_dir = path
        if cur_machine == 'goat.bu.edu':
            self.root_dir = "/scratch/chaupham/"
        if cyto_mask_path_list is None:
            cyto_mask_path_list = [os.path.join(self.root_dir, "jumpcp/BR00116991.pq")]
        # read the cyto mask df
        df = pd.concat([pd.read_parquet(path) for path in cyto_mask_path_list], ignore_index=True)
        df = self.get_split(df, split)
       

        self.data_path = list(df["path"])
        self.data_id = list(df["ID"])
        self.well_loc = list(df["well_loc"])

        assert len(perturbation_list) == 1
        self.perturbation_type = perturbation_list[0]

        if type(channels[0]) is str:
            # channel is separated by hyphen
            self.channels = torch.tensor([int(c) for c in channels[0].split("-")])
        else:
            self.channels = torch.tensor([c for c in channels])
        if scale is None and channel_mask:
            self.scale = float(self.NUM_TOTAL_CHANNELS) / len(self.channels)
        else:
            self.scale = scale  # scale the input to compensate for input channel masking

        if self.scale != 1:
            print(f"------ Scaling the input to compensate for channel masking, scale={self.scale} ------")

        print(f"------ channels: {self.channels.numpy()} ------")

        self.transform = transform

        self.plate2id, self.field2id, self.well2id, self.well2lbl = load_meta_data()

        self.channel_mask = channel_mask

    def get_split(self, df, split_name, seed=0):
        np.random.seed(seed)
        perm = np.random.permutation(df.index)
        m = len(df.index)
        train_end = int(0.6 * m)
        validate_end = int(0.2 * m) + train_end

        if split_name == "train":
            return df.iloc[perm[:train_end]]
        elif split_name == "valid":
            return df.iloc[perm[train_end:validate_end]]
        elif split_name == "test":
            return df.iloc[perm[validate_end:]]
        else:
            raise ValueError("Unknown split")

    def __getitem__(self, index):
        if self.well_loc[index] not in self.well2lbl[self.perturbation_type]:
            # this well is not labeled
            return None
        ## EDIT: use local img
        img_path = self.data_path[index].replace("s3://insitro-research-2023-context-vit/", self.root_dir)
        ## read npy img
        img_chw = np.load(img_path)  # img_chw = self.get_image(img_path)
        if img_chw is None:
            return None

        img_hwc = img_chw.transpose(1, 2, 0)
        img_chw = self.transform(img_hwc)

        channels = self.channels.numpy()

        assert type(img_chw) is not list, "Only support jumpcp for supervised training"

        if self.scale != 1:
            # scale the image pixels to compensate for the masked channels
            # used in inference
            img_chw *= self.scale

        # mask out channels
        if self.channel_mask:
            # mask out unselected channels by setting their pixel values to 0
            unselected = [c for c in range(len(img_chw)) if c not in channels]
            img_chw[unselected] = 0
        else:
            img_chw = img_chw[channels]

        return {
            "image": img_chw,
            "channels": channels,
            "label": self.well2lbl[self.perturbation_type][self.well_loc[index]],
        }

    def __len__(self) -> int:
        return len(self.data_path)

    @staticmethod
    def collate_fn(batch):
        """Filter out bad examples (None) within the batch."""
        batch = list(filter(lambda example: example is not None, batch))
        return default_collate(batch)
