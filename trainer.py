from __future__ import annotations

import collections
import os
import time
from copy import deepcopy
from os.path import join as os_join
from typing import Dict, Tuple, List, Optional
from itertools import combinations
import torchmetrics

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import defaultdict

from omegaconf import OmegaConf, ListConfig
from torch import nn, Tensor
from tqdm import tqdm
import gc
import models
import utils
from utils import get_gpu_mem
from morphem.benchmark import run_benchmark
from helper_classes.best_result import BestResult
from config import MyConfig
from datasets.dataset_utils import (
    get_channel,
    get_train_val_test_loaders,
    get_classes,
    make_random_instance_train_loader,
)
from helper_classes.channel_pooling_type import ChannelPoolingType
from helper_classes.first_layer_init import FirstLayerInit
from helper_classes.datasplit import DataSplit
from models import model_utils
from models.depthwise_convnext import DepthwiseConvNeXt
from models.hypernet_convnext import HyperConvNeXt
from models.loss_fn import proxy_loss
from models.shared_convnext import SharedConvNeXt
from lr_schedulers import create_my_scheduler
from models.slice_param_convnext import SliceParamConvNeXt
from models.template_mixing_convnext import TemplateMixingConvNeXt
from models.template_mixing_vit import TemplateMixingViT
from models.channel_vit_adapt import ChannelViTAdapt
from models.dichavit import DiChaViT
from models.vit_adapt import ViTAdapt
from models.depthwise_vit import DepthwiseViTAdapt
from models.hyper_vit import HyperViTAdapt, HyperNetViT

from optimizers import make_my_optimizer
from utils import AverageMeter, exists
from custom_log import MyLogging, DummyLogger
from models.model_utils import get_shapes, MeanEncoder, VarianceEncoder


class Trainer:
    def __init__(self, cfg: MyConfig) -> None:

        self.cfg = cfg

        self.debug = self.cfg.train.debug
        if self.debug:
            self.cfg.train.save_model = "none"

        self.use_ddp = (cfg.hardware.multi_gpus == "ddp") and torch.cuda.is_available()
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0)) if self.use_ddp else 0
        self.global_rank = int(os.environ.get("RANK", 0)) if self.use_ddp else 0
        self.acc_metric = None

        if self.use_ddp:
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = utils.get_device(self.cfg.hardware.device)

        self.shuffle_all = "SHUFFLE_ALL"

        self.data_channels = {}
        self.data_classes_train = None  ## classes of dataset, e.g., ['airplane', 'bird', ...] for CIFAR10
        self.seed = utils.default(self.cfg.train.seed, np.random.randint(1000, 1000000))
        job_id = utils.default(self.cfg.logging.scc_jobid, None)

        self.jobid_seed = f"jobid{job_id}_seed{self.seed}" if job_id is not None else f"seed{self.seed}"
        _project_name = self.cfg.logging.wandb.project_name
        self.project_name = utils.default(_project_name, "new_channels_" + self.cfg.dataset.name)

        self.all_chunks = [list(chunk.keys())[0] for chunk in self.cfg.data_chunk.chunks]
        self.cfg.eval.meta_csv_file = "enriched_meta.csv"

        self.extra_loss_lambda = self.cfg.train.extra_loss_lambda

        ## auto set eval batch size to maximize GPU memory usage
        if not self.cfg.eval.batch_size:
            if "depthwise" not in self.cfg.model.name:
                ## bs=512, takes 12 GB memory
                gpu_mem = utils.get_gpu_mem(return_total_mem=True)
                eval_batch_size = int(512 * gpu_mem / 14)
                # round to the nearest power of 2
                eval_batch_size = 2 ** int(np.log2(eval_batch_size))
            else:
                eval_batch_size = 128  ## too large will cause error

            self.cfg.eval.batch_size = eval_batch_size
            print(f"self.cfg.eval.batch_size: {self.cfg.eval.batch_size}")

        #### model = adaptive_interface + shared_part
        ## optional: train adaptive_interface only for the first few epochs
        fine_tune_lr = self.cfg.optimizer.params["lr"]
        adaptive_interface_lr = self.cfg.train.adaptive_interface_lr
        ## Default: set small lr for adaptive interface
        self.cfg.train.adaptive_interface_lr = utils.default(adaptive_interface_lr, fine_tune_lr * 100)

        #### add some info to the cfg for tracking purpose
        self.cfg.tag = utils.default(self.cfg.tag, "-".join(self.all_chunks))
        self.cfg.train.seed = self.seed
        self.cfg.logging.wandb.project_name = self.project_name

        datime_now = utils.datetime_now("%Y-%b-%d-%H-%M-%S")

        self.checkpoints = os_join(
            self.cfg.train.checkpoints,
            self.cfg.dataset.name,
            # str(DataChunk(cfg.data_chunk.chunks)),
            datime_now + "--" + self.jobid_seed,
        )

        self.mapper = None
        if self.cfg.dataset.name in ["morphem70k"]:
            if len(self.cfg.dataset.in_channel_names) == 12:
                self.mapper = {"Allen": [0, 1, 2], "HPA": [3, 4, 5, 6], "CP": [7, 8, 9, 10, 11]}
            else:
                self.mapper = {
                    "Allen": [5, 2, 6],
                    "HPA": [3, 6, 5, 0],
                    "CP": [5, 0, 7, 1, 4],
                }
            print("self.mapper", self.mapper)

        # elif self.cfg.dataset.name == "HPA":
        #     self.mapper = {"HPA": [0, 1, 2, 3]}
        elif self.cfg.dataset.name == "so2sat_city":
            self.mapper = self.cfg.data_chunk.chunks[0]["so2sat_city"]
            self.cfg.train.training_chunks = "train"
            # assert self.cfg.train.training_chunks == "train"
        elif self.cfg.dataset.name == "jump_cp":
            self.mapper = self.cfg.data_chunk.chunks[0]["jump_cp"]
            self.cfg.train.training_chunks = "train"
        else:
            raise NotImplementedError(f"dataset {self.cfg.dataset.name} not supported yet")

        self.start_epoch = 1
        self.best_model_path = os_join(self.checkpoints, "model_best.pt")
        self.last_model_path = os_join(self.checkpoints, "model_last.pt")

        self.train_loss_fn = "ce" if self.cfg.dataset.name in ["cifar10", "cifar100"] else "proxy"
        self.train_metric = "{split}_{chunk_name}/loss"
        self.train_metric_all_chunks = "{split}_ALL_CHUNKS/loss"

        self.best_res_all_chunks = collections.defaultdict(lambda: BestResult())

        self.train_loaders = {}
        self.val_loaders = {}
        self.test_loaders = {}
        self.num_loaders = None

        #### Build datasets, model, optimizer, logger
        self._build_dataset()

        if self.cfg.train.miro:  ## build MIRO, should be done before building model
            self.cfg.model.num_classes = len(self.data_classes_train)  ## duplicate, but ok for now
            if (
                hasattr(self.cfg.model, "pooling_channel_type")
                and self.cfg.model.pooling_channel_type == ChannelPoolingType.ATTENTION
            ):
                self.pre_featurizer = getattr(models, self.cfg.model.name)(
                    self.cfg.model,
                    freeze="all",
                    attn_pooling_params=self.cfg.attn_pooling,
                ).to(self.device)
                self.featurizer = getattr(models, self.cfg.model.name)(
                    self.cfg.model, attn_pooling_params=self.cfg.attn_pooling
                ).to(self.device)
            else:
                self.pre_featurizer = getattr(models, self.cfg.model.name)(self.cfg.model, freeze="all").to(
                    self.device
                )

                self.featurizer = getattr(models, self.cfg.model.name)(self.cfg.model).to(self.device)

            chunk_name = self.cfg.dataset.name
            dims = {
                "Allen": 3,
                "HPA": 4,
                "CP": 5,
                "morphem70k": 3,
            }  ## "morphem70k": 3 is placeholder, it doesn't matter for shared_MIRO
            # build mean/var encoders
            shapes = get_shapes(
                self.pre_featurizer,
                (
                    dims[chunk_name],
                    self.cfg.dataset.img_size,
                    self.cfg.dataset.img_size,
                ),
            )
            self.mean_encoders = nn.ModuleList([MeanEncoder(shape) for shape in shapes]).to(self.device)
            self.var_encoders = nn.ModuleList([VarianceEncoder(shape) for shape in shapes]).to(self.device)
        self._build_model()
        self._build_log()

        ## TODO: may make an toggle here: we don't need these for evaluation mode
        self.updates_per_epoch = len(self.train_loaders[self.shuffle_all])
        self.total_epochs_all_chunks = self.cfg.train.num_epochs
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        use_wd = self.cfg.optimizer.params.get("weight_decay_end", -1)
        if use_wd > -1:
            params_ = self.cfg.optimizer.params
            self.wd_schedule = utils.cosine_scheduler(
                params_.weight_decay,
                params_.weight_decay_end,
                self.total_epochs_all_chunks,
                self.updates_per_epoch,
            )
            print(f"---------- set up self.wd_schedule [{self.wd_schedule[0]}, {self.wd_schedule[-1]}]")
        else:
            self.wd_schedule = None
        # self.scheduler = None  ## build scheduler later in training loop

        if self.cfg.train.resume_train:
            resume_path = os_join(self.cfg.train.resume_model)
            last_epoch = self._load_model(resume_path)

            self.start_epoch = last_epoch + 1

        self.use_amp = self.cfg.train.use_amp
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)  # type: ignore

        self._log_config_and_model_info()  ## log model info to wandb

        if self.cfg.train.swa or self.cfg.train.swad:
            self.swa_model = AveragedModel(self.model)
            self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.cfg.train.swa_lr)

        return None

    @property
    def current_lr(self) -> float | List[float]:
        lr = self.optimizer.param_groups[0]["lr"]  # or self.scheduler.get_last_lr()[-1]
        ## get all lrs:
        lrs = [param_group["lr"] for param_group in self.optimizer.param_groups]  # type: ignore
        if len(set(lrs)) == 1:
            return lrs[0]
        else:
            return lrs

    @property
    def current_wd(self) -> float:
        wd = self.optimizer.param_groups[0]["weight_decay"]
        return wd

    def _get_forward_mode(self) -> str:
        """
        Forward function for different models are different.
        Some requires chunk_name as input, some don't.
        This is a helper function for _forward_model().
        """
        if self.cfg.train.training_chunks is not None:
            forward_mode = "need_training_chunks"
        elif isinstance(
            self.model,
            (
                SharedConvNeXt,
                SliceParamConvNeXt,
                TemplateMixingConvNeXt,
                TemplateMixingViT,
                HyperConvNeXt,
                DepthwiseConvNeXt,
                HyperNetViT,
                ChannelViTAdapt,
                DepthwiseViTAdapt,
                HyperViTAdapt,
                ViTAdapt,
                DiChaViT,
            ),
        ):
            forward_mode = "need_chunk_name"
        else:
            forward_mode = "normal_forward"
        return forward_mode

    def _forward_model(
        self,
        x,
        chunk_name: str | None,
        training_chunks: str | None = None,
        init_first_layer: str | None = None,
        new_channel_init: NewChannelLeaveOneOut | None = None,
        **kwargs,
    ):
        """
        forward step, depending on the type of model
        @param x:
        @param chunk_name:
        @return:
        """
        if self.forward_mode == "need_chunk_name":
            output = self.model(
                x, chunk_name, init_first_layer=init_first_layer, new_channel_init=new_channel_init, **kwargs
            )
        elif self.forward_mode == "need_training_chunks":
            output = self.model(
                x,
                chunk_name,
                training_chunks,
                init_first_layer=init_first_layer,
                new_channel_init=new_channel_init,
                **kwargs,
            )
        else:
            output = self.model(x)
        return output

    ## training loop
    def train(self):

        epoch_timer = utils.Time1Event()

        if not self.debug and not self.cfg.eval.get("skip_eval_first_epoch", False):
            self.logger.info("Before training, evaluate:")
            self.evaluate_model(epoch=0)

        num_epochs = self.cfg.train.num_epochs  ## + self.start_epoch - 1

        for epoch in range(self.start_epoch, num_epochs + 1):
            ### only train the adaptive interface for the first few epochs
            if self.cfg.train.adaptive_interface_epochs > 0:
                raise NotImplementedError("currently turn this off: 'adaptive_interface_epochs > 0'")

            ## Log
            self.logger.info(f"\n[{utils.datetime_now()}] Start Epoch {epoch}/{self.total_epochs_all_chunks}")

            ## Scheduler per epoch
            if self.scheduler and not (
                (self.cfg.train.swa or self.cfg.train.swad) and epoch > self.cfg.train.swa_start
            ):
                self.scheduler.step(epoch)

            ## train
            self.train_one_epoch(epoch, self.shuffle_all)

            ## Evaluate on ALL chunks
            if epoch % self.cfg.eval.every_n_epochs == 0 or epoch == num_epochs:
                self.evaluate_model(epoch=epoch)

            ## save cur model
            if self.local_rank == 0:
                if self.cfg.train.save_model == "all":
                    cur_model_path = os_join(self.checkpoints, f"model_{epoch}.pt")
                    self._save_model(path=cur_model_path, epoch=epoch, val_acc=None)
                elif self.cfg.train.save_model == "last":
                    self._save_model(path=self.last_model_path, epoch=epoch, val_acc=None)
                elif self.cfg.train.save_model == "best":
                    raise NotImplementedError("save_model='best' not implemented yet")
                elif self.cfg.train.save_model == "none":
                    pass
                else:
                    every_n_epoch = int(self.cfg.train.save_model)
                    if epoch % every_n_epoch == 0:
                        cur_model_path = os_join(self.checkpoints, f"model_{epoch}.pt")
                        self._save_model(path=cur_model_path, epoch=epoch, val_acc=None)

            ## Logging stuff
            epoch_timer.update()
            self.logger.info({"minute/epoch": round(epoch_timer.avg / 60, 2)})
            need_time = utils.convert_secs2time(epoch_timer.avg * (num_epochs - epoch), return_string=True)

            self.logger.info(need_time)  # type: ignore
            self.logger.info("=" * 40)
        if self.cfg.eval.eval_subset_channels:
            self.eval_subset_channels()
        self._finish_training()

    @torch.inference_mode()
    def eval_regular(self, epoch: int) -> Dict[str, float]:
        def _eval(split: str, chunk_name: str, new_channel_init: Optional[NewChannelLeaveOneOut]):
            self.model.eval()
            if split.startswith("test"):
                if isinstance(self.test_loaders.get(chunk_name), dict):
                    eval_loader = self.test_loaders.get(chunk_name).get(split)
                else:
                    eval_loader = self.test_loaders.get(chunk_name)
            elif split == "valid":
                eval_loader = self.val_loaders.get(chunk_name)

            if eval_loader is None:
                print(f"No set for split={split}, skipped!")
                return None

            self.logger.info(
                f"Start evaluation split={split}, epoch {epoch} with new_channel_init={new_channel_init}"
            )

            start_time = time.time()
            output_list = []
            gt_list = []
            training_chunks = "train" if self.mapper["train"] != self.mapper[split] else None
            for bid, batch in enumerate(eval_loader):
                if self.debug and bid > 3:
                    break
                batch = utils.move_to_cuda(batch, self.device)
                x, y, _ = batch["image"], batch["label"], batch["channels"]

                output = self._forward_model(
                    x,
                    chunk_name=split,
                    training_chunks=training_chunks,
                    init_first_layer=None,
                    new_channel_init=new_channel_init,
                )
                output_list.append(output)
                gt_list.append(y)

            output = torch.cat(output_list, dim=0)
            gt = torch.cat(gt_list, dim=0)
            accuracy = self.eval_accuracy(output, gt)

            self.logger.info(
                f"Done {split} evaluation for epoch {epoch} in {(time.time() - start_time) / 60:.2f} minutes"
            )
            if new_channel_init is None:
                self.logger.info({f"acc/{split}": accuracy})
            else:
                self.logger.info({f"acc/{split}/{new_channel_init}": accuracy})

            if self.acc_metric:
                accuracy = self.acc_metric.compute() * 100
                self.logger.info(
                    f"ALL GPUS: {split} evaluation for epoch {epoch} in {(time.time() - start_time) / 60:.2f} minutes"
                )

                if new_channel_init is None:
                    self.logger.info({f"acc_allGPUs/{split}": accuracy})
                else:
                    self.logger.info({f"acc_allGPUs/{split}/{new_channel_init}": accuracy})

                # Resetting internal state such that metric ready for new data
                self.acc_metric.reset()

            return accuracy

        if self.cfg.eval.only_eval_first_and_last:
            if epoch != 0 and epoch != self.cfg.train.num_epochs:
                return None  ## bail out, skip this expensive evaluation

        chunk_name = list(self.data_channels.keys())[0]
        res = {}
        splits = [split for split in self.mapper.keys() if split != "train"]
        for split in splits:
            for i in range(len(self.cfg.model.new_channel_inits)):
                ## if the channels are the same, only need to run 1 time
                if i > 0 and self.mapper["train"] == self.mapper[split]:
                    break
                new_channel_init = self.cfg.model.new_channel_inits[i]
                if split == "valid":
                    new_channel_init = None  ## only 1 new_channel_init for validation set

                acc = _eval(split, chunk_name, new_channel_init)

                res[f"acc/{split}/{new_channel_init}"] = acc
        return res

    @torch.inference_mode()
    def eval_subset_channels(self):
        eval_loader = self.test_loaders[self.cfg.dataset.name]
        chunk_name = "test"

        if isinstance(eval_loader, dict):
            if chunk_name not in eval_loader:
                print(f"chunk_name={chunk_name} not in eval_loader, skipped!")
                return None
            eval_loader = eval_loader[chunk_name]
        channels = self.cfg.data_chunk["chunks"][0][self.cfg.dataset.name]["train"]

        all_acc = {}
        start_time = time.time()

        self.model.eval()
        start = len(channels)
        end = 0
        for n_channels in tqdm(range(start, end, -1)):
            acc_n_channels = []
            for selected in combinations(channels, n_channels):
                selected = list(selected)
                selected = [channels.index(s) for s in selected]
                print(f"selected: {selected}")
                output_list = []
                gt_list = []
                for bid, batch in enumerate(eval_loader):
                    print(f"batch {bid}/{len(eval_loader)}")
                    batch = utils.move_to_cuda(batch, self.device)
                    x, y, _ = batch["image"], batch["label"], batch["channels"]
                    x = x[:, selected, :, :].clone()

                    if isinstance(self.model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                        model = self.model.module
                    else:
                        model = self.model

                    if hasattr(model, "mapper"):
                        model.mapper[chunk_name] = selected
                    elif hasattr(model.feature_extractor, "mapper"):
                        model.feature_extractor.mapper[chunk_name] = selected
                    elif hasattr(model.feature_extractor.patch_embed, "mapper"):
                        model.feature_extractor.patch_embed.mapper[chunk_name] = selected
                    else:
                        raise NotImplementedError("model doesn't have mapper attribute")

                    output = self._forward_model(
                        x,
                        chunk_name=chunk_name,
                        training_chunks=None,
                        init_first_layer=None,
                        new_channel_init="",
                    )
                    output_list.append(output)
                    gt_list.append(y)

                output = torch.cat(output_list, dim=0)
                gt = torch.cat(gt_list, dim=0)
                accuracy = self.eval_accuracy(output, gt)
                print(f"selected: {selected}, acc={accuracy}")
                acc_n_channels.append(accuracy)
            all_acc[n_channels] = acc_n_channels
            mean_ = torch.mean(torch.tensor(acc_n_channels)).item()
            std_ = torch.std(torch.tensor(acc_n_channels)).item()
            print(f"----- n_channels={n_channels}, acc={mean_}, std={std_}")
            self.logger.info({f"sub_channels/{n_channels}": mean_})
            break
        end_time = time.time()
        print(f"Done eval_subset_channels() in {(end_time - start_time) / 60:.2f} minutes")
        self.logger.info(str(all_acc))

        return all_acc

    def evaluate_model(self, epoch: int):
        if self.cfg.eval.only_eval_first_and_last:
            if epoch != 0 and epoch != self.cfg.train.num_epochs:
                return None  ## bail out, skip this expensive evaluation
        if self.cfg.dataset.name in ["Allen", "HPA", "CP", "morphem70k"]:
            for new_channel_init in self.cfg.model.new_channel_inits:
                self.eval_morphem70k(epoch=epoch, new_channel_init=new_channel_init)
        elif self.cfg.dataset.name in ["so2sat_city", "jump_cp"]:
            self.eval_regular(epoch=epoch)
        else:
            raise NotImplementedError(f"dataset {self.cfg.dataset.name} not supported yet")

    @torch.inference_mode()
    def eval_morphem70k(
        self, epoch: int, new_channel_init: NewChannelLeaveOneOut, eval_chunks: Optional[List[str]] = None
    ):
        def log_res(eval_cfg, knn_metric):
            call_umap = eval_cfg["umap"] and (epoch == 0 or epoch == self.cfg.train.num_epochs)
            if eval_chunks is None or len(eval_chunks) == 3:
                dataset = self.cfg.dataset.name
            else:
                dataset = eval_chunks[0]  ## TODO: for now only test on 1 transfered chunk or all chunks
            if knn_metric in ["l2", "cosine"]:
                full_res = run_benchmark(
                    eval_cfg["root_dir"],
                    eval_cfg["dest_dir"],
                    eval_cfg["feature_dir"],
                    eval_cfg["feature_file"],
                    eval_cfg["classifier"],
                    call_umap,
                    eval_cfg["use_gpu"],
                    knn_metric,
                    #dataset,  # quick hack to run benchmark on only 1 dataset
                )
                ## log results
                full_res["key"] = full_res.iloc[:, 0:3].apply(lambda x: "/".join(x.astype(str)), axis=1)
                acc = dict(
                    zip(
                        full_res["key"] + f"/{knn_metric}/acc",
                        full_res["accuracy"] * 100,
                    )
                )

                f1 = dict(
                    zip(
                        full_res["key"] + f"/{knn_metric}/f1",
                        full_res["f1_score_macro"],
                    )
                )
                metrics_logger = {
                    **acc,
                    **f1,
                    f"{classifier}/{knn_metric}/score_acc/": np.mean(list(acc.values())[1:]),
                    f"{classifier}/{knn_metric}/score_f1/": np.mean(list(f1.values())[1:]),
                }
            else:
                knn_metric = "l2"
                full_res = run_benchmark(
                    eval_cfg["root_dir"],
                    eval_cfg["dest_dir"],
                    eval_cfg["feature_dir"],
                    eval_cfg["feature_file"],
                    eval_cfg["classifier"],
                    call_umap,
                    eval_cfg["use_gpu"],
                    knn_metric,
                    #dataset,  # quick hack to run benchmark on only 1 dataset
                )
                ## log results
                full_res["key"] = full_res.iloc[:, 0:3].apply(lambda x: "/".join(x.astype(str)), axis=1)
                acc = dict(zip(full_res["key"] + f"/acc", full_res["accuracy"] * 100))
                f1 = dict(zip(full_res["key"] + f"/f1", full_res["f1_score_macro"]))
                metrics_logger = {
                    **acc,
                    **f1,
                    f"{classifier}/score_acc/": np.mean(list(acc.values())[1:]),
                    f"{classifier}/score_f1/": np.mean(list(f1.values())[1:]),
                }
            self.logger.info(metrics_logger, sep="| ", padding_space=True)
            return metrics_logger

        self.logger.info(f"Start evaluation for epoch {epoch} with new_channel_init={new_channel_init}")
        self.model.eval()

        training_chunks = self.cfg.train.training_chunks
        init_first_layer = self.cfg.model.init_first_layer

        eval_cfg = deepcopy(self.cfg.eval)
        ## make a new folder for each epoch
        scc_jobid = utils.default(self.cfg.logging.scc_jobid, "")
        FOLDER_NAME = f'{utils.datetime_now("%Y-%b-%d")}_seed{self.cfg.train.seed}_sccid{scc_jobid}'
        eval_cfg.dest_dir = os_join(eval_cfg.dest_dir.format(FOLDER_NAME=FOLDER_NAME), f"epoch_{epoch}")
        utils.mkdir(eval_cfg.dest_dir)
        utils.mkdir(eval_cfg.feature_dir.format(FOLDER_NAME=FOLDER_NAME))

        eval_cfg.feature_dir = eval_cfg.feature_dir.format(FOLDER_NAME=FOLDER_NAME)

        start_time = time.time()

        out_path_list = []
        # new_channel_init = self.cfg.model.new_channel_init
        if eval_chunks is None:
            eval_chunks = self.all_chunks
        for chunk_name in eval_chunks:
            feat_outputs = []  # store feature vectors
            eval_loader = self.test_loaders[chunk_name]
            channel_combinations = self.cfg.eval.channel_combinations

            print(f"Start getting features for {chunk_name}...")
            if channel_combinations is not None:
                print(f"======= channel_combinations: {channel_combinations}")
            total_iterations = len(eval_loader)

            for bid, batch in tqdm(enumerate(eval_loader), total=total_iterations):
                x = utils.move_to_cuda(batch, self.device)
                if channel_combinations is not None:
                    x = x[:, channel_combinations, :, :].clone()
                output = self._forward_model(
                    x,
                    chunk_name=chunk_name,
                    training_chunks=training_chunks,
                    init_first_layer=init_first_layer,
                    new_channel_init=new_channel_init,
                    # idx=bid,
                )

                if self.cfg.train.miro:
                    output = output[0]
                feat_outputs.append(output)

            torch.cuda.empty_cache()
            gc.collect()
            feat_outputs = torch.cat(feat_outputs, dim=0).cpu().numpy()

            folder_path = os_join(eval_cfg.feature_dir, chunk_name)
            utils.mkdir(folder_path)

            out_path = os_join(folder_path, eval_cfg.feature_file)
            out_path_list.append(out_path)
            utils.write_numpy(feat_outputs, out_path)
            runtime = round((time.time() - start_time) / 60, 2)
            print(f"-- Done writing features for {chunk_name} in total {runtime} minutes")

        ## after we have all features for 3 chunks (i.e., Allen, HPA, CP), we run the benchmark
        torch.cuda.empty_cache()
        cosine_metrics = None
        start = time.time()
        for classifier in eval_cfg.classifiers:
            eval_cfg.classifier = classifier

            if classifier == "knn":
                for knn_metric in eval_cfg.knn_metrics:
                    if "cosine" in knn_metric:
                        cosine_metrics = log_res(eval_cfg=eval_cfg, knn_metric=knn_metric)
                    else:
                        log_res(eval_cfg=eval_cfg, knn_metric=knn_metric)

            else:
                log_res(eval_cfg=eval_cfg, knn_metric=None)

        stop = time.time()
        print(f"Done running benchmark in {(stop - start) / 60:.2f} minutes")

        ## final_score
        if cosine_metrics and self.cfg.dataset.name == "morphem70k":
            final_scores = {}
            final_scores[f"score/allen_score/{new_channel_init}"] = cosine_metrics.get(
                "Allen/Task_two/knn/cosine/f1", 0
            )
            final_scores[f"score/hpa_score/{new_channel_init}"] = (
                cosine_metrics.get("HPA/Task_two/knn/cosine/f1", 0)
                + cosine_metrics.get("HPA/Task_three/knn/cosine/f1", 0)
            ) / 2
            final_scores[f"score/cp_score/{new_channel_init}"] = (
                cosine_metrics.get("CP/Task_two/knn/cosine/f1", 0)
                + cosine_metrics.get("CP/Task_three/knn/cosine/f1", 0)
                + cosine_metrics.get("CP/Task_four/knn/cosine/f1", 0)
            ) / 3
            final_scores[f"score/final_score/{new_channel_init}"] = (
                final_scores[f"score/allen_score/{new_channel_init}"]
                + final_scores[f"score/hpa_score/{new_channel_init}"]
                + final_scores[f"score/cp_score/{new_channel_init}"]
            ) / 3

            self.logger.info(final_scores, sep="| ", padding_space=True)

        if self.cfg.eval.clean_up:
            for out_path in out_path_list:
                os.remove(out_path)
            self.logger.info(f"cleaned up {len(out_path_list)} files after evaluation")
        return final_scores

    @torch.no_grad()
    def eval_accuracy(self, output, y):
        if self.use_ddp:
            ## new code for DDP
            acc = self.acc_metric(output, y) * 100
        else:
            pred = torch.argmax(output, dim=-1)
            correct = 100 * torch.sum(pred == y) / len(y)
            acc = correct.item()
        return acc

    def train_one_epoch(self, epoch: int, chunk_name: str):
        """
        train one epoch for `chunk_name`, chunk_name can be one of ["red", "red_green", `self.shuffle_all`, ...]
        :param epoch:
        :param chunk_name
        :return:
        """

        self.model.train()
        start = time.time()
        verbose, bid = self.cfg.train.verbose_batches, 0
        try:
            print(self.model.feature_extractor.conv1_emb.weight.norm(dim=-1))
        except:
            pass

        self.logger.info(f"Epoch {epoch} | Steps: {len(self.train_loaders[chunk_name])}")

        loss_meter = collections.defaultdict(lambda: AverageMeter())
        if self.use_ddp:
            self.train_loaders[chunk_name].sampler.set_epoch(epoch)

        for bid, batch in enumerate(self.train_loaders[chunk_name], 1):
            num_updates = (epoch - 1) * self.updates_per_epoch + bid

            ## a batch consists of images from all chunks
            if self.cfg.dataset.name in ["Allen", "HPA", "CP", "morphem70k"]:
                loss_dict = self.train_one_batch_morphem70k(batch, num_updates=num_updates, epoch=epoch)
            else:
                loss_dict = self.train_one_batch_regular(batch, num_updates=num_updates, epoch=epoch)

            for k, v in loss_dict.items():
                loss_meter[k].update(v)

            if bid % verbose == 0:
                self._update_batch_log(
                    epoch=epoch,
                    bid=bid,
                    lr=self.current_lr,
                    weight_decay=self.current_wd,
                    loss_meter=loss_meter,
                )

            if self.debug and bid > 3:
                print("Debug mode, only run 3 batches")
                break
        try:
            ## hack to deal with DataParallel
            model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            counter = model.feature_extractor.patch_embed.counter
            sorted_counter = sorted(counter.items(), key=lambda x: x[1])
            sorted_dict = {f"c_{c}": count for c, count in sorted_counter}
            self.logger.info(sorted_dict)
        except:
            pass

        if bid % verbose != 0:
            self._update_batch_log(
                epoch=epoch, bid=bid, lr=self.current_lr, weight_decay=self.current_wd, loss_meter=loss_meter
            )
        if self.cfg.train.swa and not self.cfg.train.swad and epoch > self.cfg.train.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()
        # utils.gpu_mem_report()
        self.logger.info(f"Done training epoch {epoch} in {(time.time() - start) / 60:.2f} minutes")

        if isinstance(self.model, (DepthwiseConvNeXt, DepthwiseViTAdapt)) and hasattr(
            self.model, "weighted_sum_pooling"
        ):
            for i, w_i in enumerate(self.model.weighted_sum_pooling):
                self.logger.info({f"weights/w_{i}": w_i})

        return None

    def train_one_batch_morphem70k(
        self, batch: Tuple[Dict[str, Tensor], Tensor], num_updates: int, epoch: int
    ) -> Dict:

        batch = utils.move_to_cuda(batch, self.device)

        ## Zero out grads
        self.optimizer.zero_grad()

        if self.cfg.train.training_chunks:
            all_chunks = (
                self.cfg.train.training_chunks.split("_")
                if "_" in self.cfg.train.training_chunks
                else [self.cfg.train.training_chunks]
            )
        else:
            all_chunks = self.all_chunks

        training_chunks = self.cfg.train.training_chunks
        init_first_layer = self.cfg.model.init_first_layer
        ## used when pretrained=True to init the first layer of missing channels for single model mode (or shared_Net)

        for chunk_name in all_chunks:
            ## if more than 1 chunk/dataset, and chunk_name/dataset not in this batch, skip
            if len(self.all_chunks) == 1:
                x, y = batch
            else:
                if chunk_name in batch:
                    x, y = batch[chunk_name]["image"], batch[chunk_name]["label"]
                else:
                    continue
            x = get_channel(
                self.cfg.dataset.name,
                data_channels=self.data_channels[chunk_name],
                x=x,
                device=self.device,
            )
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self._forward_model(
                    x,
                    chunk_name=chunk_name,
                    training_chunks=training_chunks,
                    init_first_layer=init_first_layer,
                    new_channel_init=None,  # ignore when training, only used in evaluation
                    cur_epoch=epoch,
                )
                if isinstance(output, tuple):
                    output, extra_loss = output
                else:
                    extra_loss = 0.0

                assert self.cfg.dataset.name in ["Allen", "HPA", "CP", "morphem70k"]
                if self.cfg.model.learnable_temp:
                    scale = self.model.logit_scale.exp()
                else:
                    try:
                        scale = self.model.scale
                    except:
                        ## dataparaallel
                        scale = self.model.module.scale

                if self.cfg.train.miro:
                    y_pred, inter_feats = output
                    loss = (
                        proxy_loss(self.model.proxies, y_pred, y, scale) + extra_loss * self.extra_loss_lambda
                    )

                    with torch.no_grad():
                        if "base" in self.cfg.model.name:
                            _, pre_feats = self.pre_featurizer(x)
                        else:
                            _, pre_feats = self.pre_featurizer(x, chunk=chunk_name)

                    reg_loss = 0.0
                    for f, pre_f, mean_enc, var_enc in model_utils.zip_strict(
                        inter_feats,
                        pre_feats,
                        self.mean_encoders,
                        self.var_encoders,
                    ):
                        # mutual information regularization
                        mean = mean_enc(f)
                        var = var_enc(f)
                        vlb = (mean - pre_f).pow(2).div(var) + var.log()
                        reg_loss += vlb.mean() / 2.0

                    loss += reg_loss * self.cfg.train.miro_ld
                else:
                    loss = (
                        proxy_loss(self.model.proxies, output, y, scale) + extra_loss * self.extra_loss_lambda
                    )
                    if self.debug:
                        print("loss", loss)
                        print("extra_loss", extra_loss)
                        print("extra_loss_lambda", self.extra_loss_lambda)

            ## scale loss then call backward to have scaled grads.
            self.scaler.scale(loss).backward()
            # loss.backward()

        ## after looping over all chunks, we call optimizer.step() once
        if exists(self.cfg.train.clip_grad_norm):
            self.scaler.unscale_(self.optimizer)  # unscale grads of optimizer
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)

        ## unscale grads of `optimizer` if it hasn't, then call optimizer.step() if grads
        # don't contain NA(s), inf(s) (o.w. skip calling)
        self.scaler.step(self.optimizer)
        # self.optimizer.step()

        ## update scaler
        self.scaler.update()

        ## Scheduler per batch
        if self.scheduler and not (self.cfg.train.swad and epoch > self.cfg.train.swa_start):
            self.scheduler.step_update(num_updates=num_updates)
            if self.wd_schedule is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if i == 0:  # only the first group is regularized
                        indx = num_updates - 1
                        if indx >= len(self.wd_schedule):
                            print("num_updates", num_updates)
                            print("len(self.wd_schedule)", len(self.wd_schedule))
                            indx = len(self.wd_schedule) - 1
                        param_group["weight_decay"] = self.wd_schedule[indx]

        loss_dict = {
            self.train_metric.format(split="TRAINING_LOSS", chunk_name=self.shuffle_all): loss.item(),
            "TRAINING_LOSS_SHUFFLE_ALL/channel_proxy_loss": (
                extra_loss.item() if isinstance(extra_loss, Tensor) else 0.0
            ),
        }  ## loss on training

        if self.cfg.train.swad and epoch > self.cfg.train.swa_start:
            self.swa_model.update_parameters(self.model)
            self.swa_scheduler.step()

        return loss_dict

    def train_one_batch_regular(
        self, batch: Tuple[Dict[str, Tensor], Tensor], num_updates: int, epoch: int
    ) -> Dict:
        assert len(self.all_chunks) == 1  ## no chunking here
        batch = utils.move_to_cuda(batch, self.device)
        x, y, _ = batch["image"], batch["label"], batch["channels"]
        ## Zero out grads
        self.optimizer.zero_grad()

        ## ignore trainng_chunks, chunk_name in this mode
        ## also, ignore auto autocast, SWA
        training_chunks = None
        chunk_name = "train"
        init_first_layer = self.cfg.model.init_first_layer
        ## used when pretrained=True to init the first layer of missing channels for single model mode (or shared_Net)
        output = self._forward_model(
            x,
            chunk_name=chunk_name,
            training_chunks=training_chunks,
            init_first_layer=init_first_layer,
            new_channel_init=None,  # ignore when training, only used in evaluation
            cur_epoch=epoch,
        )
        if isinstance(output, tuple):
            output, extra_loss = output

            if extra_loss.shape != torch.Size([]):
                if self.extra_loss_lambda > 0:
                    extra_loss = extra_loss.mean()
                else:
                    extra_loss = 0.0
            main_loss = torch.nn.CrossEntropyLoss()(output, y)
            loss = main_loss + extra_loss * self.extra_loss_lambda
        else:
            main_loss = torch.nn.CrossEntropyLoss()(output, y)
            loss = main_loss
            extra_loss = 0.0

        loss.backward()

        if exists(self.cfg.train.clip_grad_norm):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.clip_grad_norm)

        self.optimizer.step()

        ## Scheduler per batch
        if self.scheduler:
            self.scheduler.step_update(num_updates=num_updates)
            if self.wd_schedule is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if i == 0:  # only the first group is regularized
                        indx = num_updates - 1
                        if indx >= len(self.wd_schedule):
                            print("num_updates", num_updates)
                            print("len(self.wd_schedule)", len(self.wd_schedule))
                            indx = len(self.wd_schedule) - 1
                        param_group["weight_decay"] = self.wd_schedule[indx]

        loss_dict = {
            self.train_metric.format(split="TRAINING_LOSS", chunk_name=self.shuffle_all): loss.item(),
            "TRAINING_LOSS_SHUFFLE_ALL/main_loss": main_loss.item(),
            "TRAINING_LOSS_SHUFFLE_ALL/extra_loss": (
                extra_loss.item() if isinstance(extra_loss, Tensor) else 0.0
            ),
        }  ## loss on training
        return loss_dict

    def _update_batch_log(
        self,
        epoch: int,
        bid: int,
        lr: float | List[float],
        weight_decay: float,
        loss_meter: Dict,
        only_print: bool = False,
    ) -> None:
        msg_dict = {"epoch": epoch, "bid": bid}
        if isinstance(lr, list):
            lr_dict = {f"lr_{i}": lr_i for i, lr_i in enumerate(lr)}
            msg_dict.update(lr_dict)
        else:
            msg_dict["lr"] = lr
        msg_dict["weight_decay"] = weight_decay
        if self.cfg.model.learnable_temp:
            scale = self.model.logit_scale.exp()
            msg_dict["temperature"] = 1 / scale.data.item()

        for metric, value in loss_meter.items():
            msg_dict[metric] = value.avg
            value.reset()
        if only_print:
            print(msg_dict)
        else:
            self.logger.info(msg_dict)
        return None

    def _get_avg_metric_all_chunks_by_avg_chunk(
        self, key_base: str, metric: str, logger_dict: Dict, split: DataSplit
    ):
        res_list = []
        for chunk_name in self.all_chunks:
            key = key_base.format(split=split, chunk_name=chunk_name, metric=metric)
            logger = logger_dict[key]
            res_list.append(logger.avg)
        return np.mean(np.array(res_list))

    def _build_dataset(self):
        data_cfg = self.cfg.dataset
        dataset = data_cfg.name
        batch_size = self.cfg.train.batch_size
        eval_batch_size = self.cfg.eval.batch_size
        img_size = self.cfg.dataset.img_size

        num_workers = self.cfg.hardware.num_workers
        data_chunks = self.cfg.data_chunk.chunks

        root_dir = data_cfg.root_dir
        file_name = data_cfg.file_name
        tps_prob = self.cfg.train.tps_prob
        ssl_flag = self.cfg.train.ssl

        channels = self.mapper if dataset in ["so2sat_city", "jump_cp"] else None

        for chunk in data_chunks:
            chunk_name = list(chunk.keys())[0]
            train_loader, val_loader, test_loader = get_train_val_test_loaders(
                use_ddp=self.use_ddp,
                dataset=dataset,
                img_size=img_size,
                chunk_name=chunk_name,
                seed=self.seed,
                batch_size=batch_size,
                eval_batch_size=eval_batch_size,
                num_workers=num_workers,
                root_dir=root_dir,
                file_name=file_name,
                tps_prob=tps_prob,
                ssl_flag=ssl_flag,
                channels=channels,
            )

            self.train_loaders[chunk_name] = train_loader
            self.val_loaders[chunk_name] = val_loader
            if isinstance(test_loader, DataLoader):
                self.test_loaders[chunk_name] = test_loader
            elif isinstance(test_loader, dict):
                self.test_loaders[chunk_name] = {}
                self.test_loaders[chunk_name].update(test_loader)
            else:
                raise ValueError(f"test_loader is not valid: {test_loader}")
            self.data_channels[chunk_name] = chunk[chunk_name]

        training_chunks = self.cfg.train.training_chunks

        train_loader_all = make_random_instance_train_loader(
            dataset,
            img_size,
            batch_size=batch_size,
            seed=self.seed,
            num_workers=num_workers,
            root_dir=root_dir,
            file_name=file_name,
            tps_prob=tps_prob,
            ssl_flag=ssl_flag,
            training_chunks=training_chunks,
        )
        self.train_loaders[self.shuffle_all] = utils.default(train_loader_all, train_loader)

        self.num_loaders = len(data_chunks)
        training_chunks_list = training_chunks.split("_") if training_chunks else None
        self.data_classes_train, self.data_classes_test = get_classes(
            dataset, file_name, training_chunks_list
        )  ##

    def _build_model(self):
        self.cfg.model.in_channel_names = self.cfg.dataset.in_channel_names
        ## force add "img_size" to model
        if "img_size" not in self.cfg["model"]:
            OmegaConf.update(self.cfg, "model.img_size", [self.cfg.dataset.img_size], force_add=True)
        assert self.data_classes_train is not None, "self.data_classes_train is None!"
        self.cfg.model.num_classes = len(self.data_classes_train)

        if self.cfg.train.miro:
            self.model = self.featurizer

        else:
            if (
                hasattr(self.cfg.model, "pooling_channel_type")
                and self.cfg.model.pooling_channel_type == ChannelPoolingType.ATTENTION
            ):
                if "miro" in self.cfg.model.name:
                    self.model = getattr(models, self.cfg.model.name)(
                        self.cfg.model,
                        freeze=None,
                        attn_pooling_params=self.cfg.attn_pooling,
                    )
                else:
                    self.model = getattr(models, self.cfg.model.name)(
                        self.cfg.model, attn_pooling_params=self.cfg.attn_pooling, mapper=self.mapper
                    )
            else:
                self.model = getattr(models, self.cfg.model.name)(self.cfg.model, mapper=self.mapper)

        self.model = self.model.to(self.device)
        ## check torch version >= 2
        self.forward_mode = self._get_forward_mode()  # determine the type of self.model before compiling
        if torch.__version__ >= "2.0.0" and self.cfg.train.get("compile_pytorch", False):
            self.model = torch.compile(self.model, mode="reduce-overhead")

        if self.cfg.hardware.multi_gpus == "DataParallel":
            print("os.environ['CUDA_VISIBLE_DEVICES']", os.getenv("CUDA_VISIBLE_DEVICES"))
            print(f"using {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(self.model)
        elif self.cfg.hardware.multi_gpus == "ddp":  ## self.use_ddp
            print("os.environ['CUDA_VISIBLE_DEVICES']", os.getenv("CUDA_VISIBLE_DEVICES"))
            ## compute accuracy for DDP
            if self.cfg.dataset.name not in ["Allen", "HPA", "CP", "morphem70k"]:
                self.acc_metric = torchmetrics.classification.Accuracy(
                    task="multiclass", num_classes=len(self.data_classes_test)
                )
                self.acc_metric = self.acc_metric.to(self.device)

            self.model = DDP(self.model, device_ids=[self.local_rank], find_unused_parameters=True)

        elif self.cfg.hardware.multi_gpus is None:
            pass
        else:
            raise ValueError(f"{self.cfg.hardware.multi_gpus} is not valid!")

    def _build_log(self):
        if self.use_ddp and self.local_rank != 0:
            print("DummyLogger for self.local_rank", self.local_rank)
            self.logger = DummyLogger(self.cfg)
        else:
            self.logger = MyLogging(
                self.cfg,
                model=self.model,
                job_id=self.jobid_seed,
                project_name=self.project_name,
            )

    def _log_config_and_model_info(self):
        if self.cfg.logging.wandb.use_wandb:
            self.logger.log_config(self.cfg)

        self.logger.info(OmegaConf.to_yaml(self.cfg))
        if self.use_ddp and self.local_rank != 0:
            pass
        else:
            self.logger.info(str(self.model))
            total_num, trainable_num = utils.analyze_model(self.model, print_trainable=True)
            self.logger.info({"total_params": total_num, "trainable_params": trainable_num})
            self.logger.info("Pytorch version: {}".format(torch.__version__))
            self.logger.info("Cuda version: {}".format(torch.version.cuda))  # type: ignore
            self.logger.info("Cudnn version: {}".format(torch.backends.cudnn.version()))  # type: ignore

    def _build_optimizer(self):
        name = self.cfg.optimizer.name
        optimizer_cfg = dict(**self.cfg.optimizer.params)

        ## get all trainable params
        all_trainable_params = [p for p in self.model.parameters() if p.requires_grad]

        lr = optimizer_cfg["lr"]
        miro_lr_mult = self.cfg.train.miro_lr_mult
        param_list = [
            {"params": all_trainable_params, "lr": lr},
        ]

        if self.cfg.train.miro:
            miro_params = [
                {"params": self.mean_encoders.parameters(), "lr": lr * miro_lr_mult},
                {
                    "params": self.var_encoders.parameters(),
                    "lr": lr * miro_lr_mult,
                },
            ]
            param_list += miro_params

        optimizer = make_my_optimizer(name, param_list, optimizer_cfg)

        total_params_in_model, trainable_params_in_model = utils.analyze_model(
            self.model, print_trainable=False, verbose=False
        )

        ## all trainable params in the optimizer
        total_params_in_optimizer = sum(p.numel() for p in optimizer.param_groups[0]["params"])

        if (
            trainable_params_in_model != total_params_in_model
            or total_params_in_optimizer != total_params_in_model
        ):
            print(
                "--------------- Check on the #parameters:"
                f"#trainable_params_in_model = {trainable_params_in_model}",
                f"total_params_in_model = {total_params_in_model}",
                f"total_params_in_optimizer = {total_params_in_optimizer}",
            )
        return optimizer

    def _build_scheduler(self):
        # https://github.com/rwightman/pytorch-image-models/blob/9f5bba9ef9db8a32a5a04325c8eb181c9f13a9b2/timm/scheduler/scheduler_factory.py
        sched_name = self.cfg.scheduler.name.lower()
        if sched_name == "none":
            ## bail out if no scheduler
            return None
        sched_cfg = dict(**self.cfg.scheduler.params)
        sched_cfg["t_initial"] = self.cfg.train.num_epochs - self.cfg.train.adaptive_interface_epochs
        t_in_epochs = sched_cfg.get("t_in_epochs", True)
        convert_to_batch = self.cfg.scheduler.convert_to_batch
        if convert_to_batch and not t_in_epochs:
            for k in sched_cfg:
                if k in ["t_initial", "warmup_t", "decay_t"]:
                    if isinstance(sched_cfg[k], ListConfig):
                        sched_cfg[k] = (np.array(sched_cfg[k]) * self.updates_per_epoch).tolist()
                    else:
                        sched_cfg[k] = sched_cfg[k] * self.updates_per_epoch
        scheduler = create_my_scheduler(self.optimizer, sched_name, sched_cfg)

        self.logger.info(
            {
                "updates_per_epoch": self.updates_per_epoch,
                "total_epochs_all_chunks": self.total_epochs_all_chunks,
            }
        )

        self.logger.info(scheduler.state_dict())
        return scheduler

    def _save_model(self, path: str, epoch: int, val_acc: float | None):
        utils.mkdir(self.checkpoints)
        state_dict = {
            "epoch": epoch,
            "accuracy": val_acc,
            "config": self.cfg,
            "optimizer_params": self.optimizer.state_dict(),
            "model_params": self.model.state_dict(),
            "scheduler_params": self.scheduler.state_dict() if exists(self.scheduler) else None,
            "scaler_params": self.scaler.state_dict(),
            "datetime": utils.datetime_now(),
        }

        torch.save(state_dict, path)
        self.logger.info(f"saved model to {path}")

    def _load_model(self, path):
        loc = f"cuda:{self.local_rank}"
        state_dict = torch.load(path, map_location=loc)

        num_gpus = torch.cuda.device_count()
        if num_gpus == 1 and list(state_dict["model_params"].keys())[0].startswith("module"):
            self.model.load_state_dict(
                {k.replace("module.", ""): v for k, v in state_dict["model_params"].items()}
            )
            print("loaded model from DataParallel on 1 GPU!")
        else:
            self.model.load_state_dict(state_dict["model_params"])
        self.optimizer.load_state_dict(state_dict["optimizer_params"])
        if "scheduler_params" in state_dict:
            self.scheduler.load_state_dict(state_dict["scheduler_params"])
        # self.scaler.load_state_dict(state_dict["scaler_params"])

        epoch = int(state_dict.get("epoch", 0))
        self.logger.info("loaded model from {path}, epoch {epoch}".format(path=path, epoch=epoch))

        return epoch

    def _finish_training(self):
        best_res = self.best_res_all_chunks[DataSplit.TEST]

        ## Log the best model
        if self.cfg.train.swa or self.cfg.train.swad:
            torch.optim.swa_utils.update_bn(self.train_loaders[self.shuffle_all], self.swa_model)
            self.model = self.swa_model

        self.logger.info(best_res.to_dict(), use_wandb=False, sep="| ", padding_space=True)
        h = w = int(self.cfg.dataset.img_size)
        last_model_path = self.last_model_path if self.cfg.train.save_model == "last" else ""
        self.logger.finish(
            msg_str="--------------- DONE TRAINING! ---------------",
            model=self.model,
            model_name=last_model_path,
        )

        return None
