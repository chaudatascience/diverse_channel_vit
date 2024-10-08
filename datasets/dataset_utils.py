from __future__ import annotations

import os
from typing import Tuple, List, Dict, Optional
import torch
from torch import Tensor
import torchvision
from torch.utils.data import SubsetRandomSampler, DataLoader, Dataset
from torchvision.transforms import transforms
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
import utils
from utils import default
from datasets.morphem70k import SingleCellDataset
from datasets.so2sat import So2Sat
from datasets.tps_transform import TPSTransform, dummy_transform
from datasets.jump_cp_transforms import CellAugmentation
from datasets.jump_cp import JUMPCP


class So2SatAugmentation(object):
    def __init__(
        self,
        is_train: bool,
        normalization_mean: list[float] = [0.4914, 0.4822, 0.4465],
        normalization_std: list[float] = [0.2023, 0.1994, 0.2010],
        channel_mask=[],
    ):
        self.mean = np.array([m for m in normalization_mean])[:, np.newaxis, np.newaxis]
        self.std = np.array([m for m in normalization_std])[:, np.newaxis, np.newaxis]

        self.is_train = is_train
        self.channel_mask = list(channel_mask)

    def __call__(self, img) -> Union[list[torch.Tensor], torch.Tensor]:
        """
        Take a PIL image, generate its data augmented version
        """
        # print(
        #     "img b", img[0].mean(), img[1].mean(), img[2].mean(), img[3].mean(), img[4].mean(), img[5].mean()
        # )
        img = (img - self.mean) / self.std
        # print(
        #     "img a", img[0].mean(), img[1].mean(), img[2].mean(), img[3].mean(), img[4].mean(), img[5].mean()
        # )
        # print("\n")

        if self.is_train:
            # rotation
            r = random.randint(0, 3)
            img = np.rot90(img, r, (1, 2))

            # flip
            f = random.randint(0, 1)
            if f == 1:
                img = np.flip(img, 1)

            # flip
            f = random.randint(0, 1)
            if f == 1:
                img = np.flip(img, 2)

        if len(self.channel_mask) == 0:
            # do not mask channels
            return img
        else:
            # mask out the channels
            # NOTE: this channel mask index is relative / not absolute.
            # For instance, in JUMPCP where we have 8 channels.
            # If the data loader only sends over 3-channel images with channel 5, 6, 7.
            # The channel mask should be [0] if we want to mask out 5.
            img[self.channel_mask, :, :] = 0

            return img


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return torch.stack([self.base_transform(x) for i in range(self.n_views)], dim=0)


def get_mean_std_dataset(dataset):
    """Calculate mean and std of cifar10, cifar100"""
    mean_cifar10, std_cifar10 = [0.49139968, 0.48215841, 0.44653091], [
        0.24703223,
        0.24348513,
        0.26158784,
    ]
    mean_cifar100, std_cifar100 = [0.50707516, 0.48654887, 0.44091784], [
        0.26733429,
        0.25643846,
        0.27615047,
    ]

    ## mean, std on training sets
    mean_allen, std_allen = [0.17299628, 0.21203272, 0.06717163], [
        0.31244728,
        0.33736905,
        0.15192129,
    ]
    mean_hpa, std_hpa = (
        [0.08290479, 0.041127298, 0.064044416, 0.08445485],
        [0.16213107, 0.1055938, 0.17713426, 0.1631108],
    )
    mean_cp, std_cp = (
        [0.09957531, 0.19229747, 0.16250895, 0.1824028, 0.14978175],
        [0.1728119, 0.16629605, 0.15171643, 0.14863704, 0.1524553],
    )

    ## so2sat city split
    mean_so2sat_city = [
        -3.5912242e-05,
        -7.658551e-06,
        5.937501e-05,
        2.516598e-05,
        0.044198506,
        0.25761467,
        0.0007556685,
        0.0013503395,
        0.12375654,
        0.109277464,
        0.101086065,
        0.114239536,
        0.15926327,
        0.18147452,
        0.17457514,
        0.1950194,
        0.15428114,
        0.109052904,
    ]
    std_so2sat_city = [
        0.17555329,
        0.17556609,
        0.4599934,
        0.45599362,
        2.855352,
        8.322579,
        2.44937,
        1.464371,
        0.0395863,
        0.047778852,
        0.066362865,
        0.063593246,
        0.07744504,
        0.09099384,
        0.09217117,
        0.10162713,
        0.09989747,
        0.0877891,
    ]

    mean_jump_cp = [
        4.031743599139058,
        1.565935237087539,
        3.77367898215863,
        3.4605251427133257,
        4.1723172504050225,
        6.780529773318951,
        6.787385700135139,
        6.778120829362721,
    ]
    std_jump_cp = [
        17.318438884455695,
        12.015918256263747,
        16.966058078452495,
        15.064776266287147,
        17.964118200870608,
        21.638766346725316,
        21.670565699654457,
        21.639488585095584,
    ]

    if dataset == "cifar10":
        return mean_cifar10, std_cifar10
    elif dataset == "cifar100":
        return mean_cifar100, std_cifar100
    elif dataset == "Allen":
        return mean_allen, std_allen
    elif dataset == "CP":
        return mean_cp, std_cp
    elif dataset == "HPA":
        return mean_hpa, std_hpa
    elif dataset == "morphem70k":
        return {
            "CP": (mean_cp, std_cp),
            "Allen": (mean_allen, std_allen),
            "HPA": (mean_hpa, std_hpa),
        }
    elif dataset == "so2sat_city":
        return mean_so2sat_city, std_so2sat_city
    elif dataset == "jump_cp":
        return mean_jump_cp, std_jump_cp
    else:
        raise ValueError()


def get_data_transform(dataset: str, img_size: int, tps_prob: float, ssl_flag: bool):
    """
    if tps_prob > 0, then apply TPS transform with probability tps_prob
    """
    if dataset != "morphem70k":
        mean_data, std_data = get_mean_std_dataset(dataset)

    no_transform = transforms.Lambda(dummy_transform)

    if dataset in ["cifar10", "cifar100"]:
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(img_size, padding=4),
                transforms.Resize(img_size),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean_data, std_data),
            ]
        )

        transform_eval = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean_data, std_data),
            ]
        )
    elif dataset in ["Allen", "CP", "HPA"]:
        transform_train = transforms.Compose(
            [
                TPSTransform(p=tps_prob) if tps_prob > 0 else no_transform,
                transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomRotation(15),
                # transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                ## transforms.ToTensor(), input is already a Tensor
                transforms.Normalize(mean_data, std_data),
            ]
        )

        transform_eval = transforms.Compose(
            [
                transforms.Resize(img_size, antialias=True),
                transforms.CenterCrop(img_size),
                transforms.Normalize(mean_data, std_data),
            ]
        )

    elif dataset == "morphem70k":
        mean_stds = get_mean_std_dataset(dataset)
        transform_train = {}
        transform_eval = {}
        for data in ["CP", "Allen", "HPA"]:
            mean_data, std_data = mean_stds[data]
            transform_train_ = transforms.Compose(
                [
                    TPSTransform(p=tps_prob) if tps_prob > 0 else no_transform,
                    transforms.RandomResizedCrop(
                        img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize(mean_data, std_data),
                ]
            )

            transform_eval_ = transforms.Compose(
                [
                    transforms.Resize(img_size, antialias=True),
                    transforms.CenterCrop(img_size),
                    transforms.Normalize(mean_data, std_data),
                ]
            )

            transform_train[data] = transform_train_
            transform_eval[data] = transform_eval_
    elif dataset == "so2sat_city":
        transform_train = So2SatAugmentation(
            is_train=True, normalization_mean=mean_data, normalization_std=std_data
        )

        transform_eval = So2SatAugmentation(
            is_train=False, normalization_mean=mean_data, normalization_std=std_data
        )
    elif dataset == "jump_cp":
        transform_train = CellAugmentation(
            is_train=True, normalization_mean=mean_data, normalization_std=std_data
        )

        transform_eval = CellAugmentation(
            is_train=False, normalization_mean=mean_data, normalization_std=std_data
        )
    else:
        raise ValueError(f"dataset `{dataset}` not valid!")
    if ssl_flag:
        if dataset in ["Allen", "CP", "HPA"]:
            transform_train = ContrastiveLearningViewGenerator(transform_train, n_views=2)
        else:
            for data in ["CP", "Allen", "HPA"]:
                transform_train[data] = ContrastiveLearningViewGenerator(transform_train[data], n_views=2)

    return transform_train, transform_eval


def get_in_dim(chunks: List[Dict]) -> List[int]:
    print("chunks", chunks)
    channels = [len(list(c.values())[0]) for c in chunks]
    print("channels", channels)
    return channels


def get_channel(dataset: str, data_channels: List[str], x: Tensor, device) -> Tensor:
    if dataset in ["cifar10", "cifar100"]:
        return _get_channel_cifar(data_channels, x, device)
    elif dataset in ["Allen", "CP", "HPA", "morphem70k"]:
        return x
    else:
        raise NotImplementedError()


def _get_channel_cifar(data_channels: List[str], x: Tensor, device) -> Tensor:
    """x: batch of images, shape b, c, h, w, order of colors are RGB"""
    mapper = {"red": 0, "green": 1, "blue": 2}
    NUM_CHANNELS = 3
    ALL_CHANNELS = sorted(["red", "green", "blue"])

    assert len(data_channels) <= NUM_CHANNELS
    if sorted(data_channels) == ALL_CHANNELS:
        return x

    out = []

    # example for `data_channels`: data_channels = ["red", "red_green", "ZERO"]

    b, c, h, w = x.shape
    for channel in data_channels:
        if channel in mapper:  ## either red, green, or blue
            c_idx = mapper[channel]
            out.append(x[:, c_idx : c_idx + 1, ...])
        else:  # avg of some channels, or fill by zero
            splits = channel.split("_")
            if len(splits) > 1:
                reduce, channel_list = channel.split("_")[0].lower(), channel.split("_")[1:]
            else:
                reduce, channel_list = channel.split("_")[0].lower(), []

            if reduce == "avg":
                c_idx_list = [mapper[c] for c in channel_list]
                out.append(x[:, c_idx_list, ...].mean(dim=1, keepdim=True))
            elif reduce == "zero":
                out.append(torch.zeros(b, 1, h, w, device=device))
            else:
                raise ValueError()

    res = torch.concat(out, dim=1)
    return res


def get_samplers(dataset: str, img_size: int, chunk_name: str, train_split: bool) -> Tuple:
    def get_sampler_helper() -> SubsetRandomSampler:
        """
        For CIFAR, we split the dataset into 3 smaller dataset: only red, red_green, green_blue with equal size
        return indices for the sub-datasets
        :return:
        """
        ## we split dataset into 3 smaller ones by using datasets.split_datasets
        ## Read the indices for each dataset back
        if dataset in ["cifar10", "cifar100"]:
            split = "train" if train_split else "test"
            indices = utils.read_json(f"data/split/{dataset}_{split}.json")
            data_channel_idx = f"{chunk_name}_idx"
            sampler = SubsetRandomSampler(indices[data_channel_idx])
            return sampler
        else:
            raise ValueError()

    transform_train, transform_eval = get_data_transform(dataset, img_size, tps_prob=0.0, ssl_flag=False)

    if dataset in ["cifar10", "cifar100"]:
        torch_dataset = getattr(torchvision.datasets, dataset.upper())
        data_set = torch_dataset(root="./data", train=train_split, download=True, transform=transform_train)
        data_sampler = get_sampler_helper()
        return data_set, data_sampler
    else:
        raise ValueError()


def get_train_val_test_loaders(
    use_ddp: bool,
    dataset: str,
    img_size: int,
    chunk_name: str,
    seed: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: int,
    root_dir: str,
    file_name: str,
    tps_prob: float,
    ssl_flag: bool,
    channels: Optional[Dict] = None,  ## used for so2sat_city, jump_cp
    **kwargs,
) -> Tuple[DataLoader, DataLoader | None, DataLoader | Dict[str, DataLoader]]:
    if use_ddp and dataset not in ["so2sat_city", "jump_cp"]:
        raise ValueError(f"dataset {dataset} not fixed to use with DDP")

    train_loader, val_loader, test_loader = None, None, None
    if dataset in ["cifar10", "cifar100"]:
        train_set, train_sampler = get_samplers(
            dataset, img_size=img_size, chunk_name=chunk_name, train_split=True
        )
        eval_set, eval_sampler = get_samplers(
            dataset, img_size=img_size, chunk_name=chunk_name, train_split=False
        )

        utils.set_seeds(seed + 24122022)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            pin_memory=True,
            drop_last=True,
        )

        utils.set_seeds(seed + 25122022)
        test_loader = DataLoader(
            eval_set,
            batch_size=eval_batch_size,
            sampler=eval_sampler,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            pin_memory=True,
        )
    elif dataset in ["Allen", "CP", "HPA", "morphem70k"]:
        csv_path = os.path.join(root_dir, file_name)

        transform_train, transform_eval = get_data_transform(
            chunk_name, img_size, tps_prob=tps_prob, ssl_flag=ssl_flag
        )
        train_set = SingleCellDataset(
            csv_path,
            chunk_name,
            root_dir,
            is_train=True,
            ssl_flag=ssl_flag,
            transform=transform_train,
        )

        test_set = SingleCellDataset(
            csv_path,
            chunk_name,
            root_dir,
            is_train=False,
            ssl_flag=ssl_flag,
            transform=transform_eval,
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            pin_memory=True,
            drop_last=True,
        )

        ## IMPORTANT: set shuffle to False for test set. Otherwise, the order of the test set will be different when evaluating
        test_loader = DataLoader(
            test_set,
            batch_size=eval_batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            pin_memory=True,
        )
    elif dataset == "so2sat_city":
        transform_train, transform_eval = get_data_transform(  ## TODO: add tsp_prob and ssl_flag
            dataset, img_size, tps_prob=False, ssl_flag=False
        )
        train_set = So2Sat(
            path=root_dir, transform=transform_train, channels=channels["train"], split="train"
        )
        if channels["valid"] is not None:
            valid_set = So2Sat(
                path=root_dir, transform=transform_eval, channels=channels["valid"], split="valid"
            )
            utils.set_seeds(seed + 21022024)
            val_loader = DataLoader(
                valid_set,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=utils.worker_init_fn,
                pin_memory=True,
                sampler=DistributedSampler(valid_set) if use_ddp else None,
            )

        test_keys = sorted([k for k in channels.keys() if k.startswith("test")])
        test_loader_dict = {}
        for test_key in test_keys:
            test_set = So2Sat(
                path=root_dir, transform=transform_eval, channels=channels[test_key], split="test"
            )
            utils.set_seeds(seed + 21022025)
            test_loader = DataLoader(
                test_set,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=utils.worker_init_fn,
                pin_memory=True,
                sampler=DistributedSampler(test_set) if use_ddp else None,
            )
            test_loader_dict[test_key] = test_loader
        if len(test_keys) == 1:
            test_loader = test_loader_dict[test_keys[0]]
        else:
            test_loader = test_loader_dict

        utils.set_seeds(seed + 21022023)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False if use_ddp else True,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_set) if use_ddp else None,
        )
    elif dataset == "jump_cp":
        transform_train, transform_eval = get_data_transform(  ## TODO: add tsp_prob and ssl_flag
            dataset, img_size, tps_prob=False, ssl_flag=False
        )

        train_set = JUMPCP(
            path=root_dir, split="train", transform=transform_train, channels=channels["train"]
        )
        if channels["valid"] is not None:
            valid_set = JUMPCP(
                path=root_dir,
                split="valid",
                transform=transform_eval,
                channels=channels["valid"],
            )
            utils.set_seeds(seed + 21022024)
            val_loader = DataLoader(
                valid_set,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=utils.worker_init_fn,
                pin_memory=True,
                sampler=DistributedSampler(valid_set) if use_ddp else None,
            )

        test_keys = sorted([k for k in channels.keys() if k.startswith("test")])
        test_loader_dict = {}
        for test_key in test_keys:
            test_set = JUMPCP(
                path=root_dir,
                split="test",
                transform=transform_eval,
                channels=channels[test_key],
            )
            utils.set_seeds(seed + 21022025)
            test_loader = DataLoader(
                test_set,
                batch_size=eval_batch_size,
                shuffle=False,
                num_workers=num_workers,
                worker_init_fn=utils.worker_init_fn,
                pin_memory=True,
                sampler=DistributedSampler(test_set) if use_ddp else None,
            )
            test_loader_dict[test_key] = test_loader
        if len(test_keys) == 1:
            test_loader = test_loader_dict[test_keys[0]]
        else:
            test_loader = test_loader_dict

        utils.set_seeds(seed + 21022023)
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False if use_ddp else True,
            num_workers=num_workers,
            worker_init_fn=utils.worker_init_fn,
            pin_memory=True,
            drop_last=True,
            sampler=DistributedSampler(train_set) if use_ddp else None,
        )
    else:
        raise ValueError(f"dataset {dataset} not valid!")

    return train_loader, val_loader, test_loader


def get_classes(dataset: str, file_name: str, training_chunks: List[str] | None = None) -> Tuple:
    if dataset in ["cifar10", "cifar100"]:
        torch_dataset = getattr(torchvision.datasets, dataset.upper())
        train_set = torch_dataset(root="./data", train=True, download=True, transform=None)
        train_classes = test_classes = train_set.classes

    elif dataset in ["Allen", "CP", "HPA", "morphem70k"]:
        allen_v2 = ["M0", "M1M2", "M3", "M4M5", "M6M7_complete", "M6M7_single"]
        allen_v1 = ["Interphase", "Mitotic"]
        allen = allen_v2 if "morphem70k_v2" in file_name else allen_v1
        hpa = ["golgi apparatus", "microtubules", "mitochondria", "nuclear speckles"]
        cp = ["BRD-A29260609", "BRD-K04185004", "BRD-K21680192", "DMSO"]

        if dataset == "Allen":
            train_classes = allen
        elif dataset == "HPA":
            train_classes = hpa
        elif dataset == "CP":
            train_classes = cp
        elif dataset == "morphem70k":
            if training_chunks is None:
                train_classes = allen + hpa + cp
            else:
                train_classes = []
                if "Allen" in training_chunks:
                    train_classes += allen
                if "HPA" in training_chunks:
                    train_classes += hpa
                if "CP" in training_chunks:
                    train_classes += cp
        else:
            raise NotImplementedError(f"dataset {dataset} not valid!")
        test_classes = None
    elif dataset == "so2sat_city":
        test_classes = train_classes = list(range(17))
    elif dataset == "jump_cp":
        test_classes = train_classes = list(range(161))
    else:
        raise ValueError(f"dataset {dataset} not valid!")

    return train_classes, test_classes


def make_cifar_random_instance_train_loader(
    dataset: str, img_size: int, batch_size: int, seed: int, num_workers: int
) -> DataLoader:
    transform_train, _ = transform_train, _ = get_data_transform(
        dataset, img_size, tps_prob=0.0, ssl_flag=False
    )

    train_set = CifarRandomInstance(dataset, transform_train)

    utils.set_seeds(seed + 2052023)
    cifar_collate = get_collate(CifarRandomInstance)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=cifar_collate,
        num_workers=num_workers,
        shuffle=True,
        drop_last=True,
    )
    return train_loader


def make_morphem70k_random_instance_train_loader(
    img_size: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    root_dir: str,
    file_name: str,
    tps_prob: float,
    ssl_flag: bool,
    training_chunks: str | None = None,
) -> DataLoader:
    csv_path = os.path.join(root_dir, file_name)
    dataset = "morphem70k"
    training_chunks = default(training_chunks, dataset)
    transform_train, _ = get_data_transform(dataset, img_size, tps_prob=tps_prob, ssl_flag=ssl_flag)
    train_set = SingleCellDataset(
        csv_path,
        chunk=training_chunks,  # type: ignore
        root_dir=root_dir,
        is_train=True,
        ssl_flag=ssl_flag,
        transform=transform_train,
    )

    utils.set_seeds(seed + 20230322)
    if training_chunks == "morphem70k":
        training_chunks = ["Allen", "HPA", "CP"]
    else:
        training_chunks = training_chunks.split("_")

    if dataset == "morphem70k":
        morphem_collate = get_collate(training_chunks)
    else:
        morphem_collate = None

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        collate_fn=morphem_collate,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=utils.worker_init_fn,
        pin_memory=True,
        drop_last=True,
    )
    return train_loader


def make_random_instance_train_loader(
    dataset: str,
    img_size: int,
    batch_size: int,
    seed: int,
    num_workers: int,
    root_dir: str,
    file_name: str,
    tps_prob: float,
    ssl_flag: bool,
    training_chunks: str,
) -> DataLoader:
    if dataset in ["cifar10", "cifar100"]:
        return make_cifar_random_instance_train_loader(dataset, img_size, batch_size, seed, num_workers)
    elif dataset in ["morphem70k"]:
        return make_morphem70k_random_instance_train_loader(
            img_size,
            batch_size,
            seed,
            num_workers,
            root_dir,
            file_name,
            tps_prob=tps_prob,
            ssl_flag=ssl_flag,
            training_chunks=training_chunks,
        )
    else:
        return None


# def get_collate(class_name: Dataset):
#     """
#     class_name: one of  [CifarRandomInstance, SingleCellDataset]
#     """

#     def collate(data):
#         out = {chunk: {"image": [], "label": []} for chunk in class_name.chunk_names}

#         for d in data:
#             out[d["chunk"]]["image"].append(d["image"])
#             out[d["chunk"]]["label"].append(d["label"])
#         for chunk in out:
#             out[chunk]["image"] = torch.stack(out[chunk]["image"], dim=0)
#             if out[chunk]["image"].dim() > 4:
#                 out[chunk]["image"] = torch.flatten(out[chunk]["image"], end_dim=1)
#                 out[chunk]["label"] = torch.tensor(out[chunk]["label"]).repeat_interleave(2)
#             else:
#                 out[chunk]["label"] = torch.tensor(out[chunk]["label"])
#         return out

#     return collate


def get_collate(chunk_names: List[str]):
    def collate(data):
        out = {chunk: {"image": [], "label": []} for chunk in chunk_names}
        for d in data:
            out[d["chunk"]]["image"].append(d["image"])
            out[d["chunk"]]["label"].append(d["label"])
        for chunk in out:
            out[chunk]["image"] = torch.stack(out[chunk]["image"], dim=0)
            out[chunk]["label"] = torch.tensor(out[chunk]["label"])
        return out

    return collate
