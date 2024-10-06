from typing import Optional, List

import numpy as np
from einops import rearrange
from timm import create_model
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import gc
import random

from config import Model, AttentionPoolingParams
from helper_classes.channel_pooling_type import ChannelPoolingType
from helper_classes.feature_pooling import FeaturePooling
from models.channel_attention_pooling import ChannelAttentionPoolingLayer
from models.model_utils import conv1x1
from helper_classes.first_layer_init import NewChannelLeaveOneOut


def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)


class LayerNorm2d(nn.LayerNorm):
    r"""LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/models/convnext.py
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        if _is_contiguous(x):
            return F.layer_norm(
                x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps
            ).permute(0, 3, 1, 2)
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class FanAttentionV2(nn.Module):
    def __init__(self, emb_dim, mlp_dropout=0.1):
        super().__init__()

        # self.WQ = nn.Sequential(nn.Softmax(dim=-1), nn.Linear(emb_dim, emb_dim))
        self.WQ = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 16), nn.ReLU(), nn.Linear(emb_dim // 16, emb_dim)
        )

    def forward(self, x):
        """
        x: b, c, d
        attn: b, c
        """
        queries = self.WQ(x)  # b, c, h
        keys = x.mean(dim=1)  # b, h

        sim = torch.einsum("bch, bh -> bc", queries, keys)
        attn = F.sigmoid(sim)
        return attn


class DepthwiseConvNeXt(nn.Module):
    def __init__(
        self,
        config: Model,
        attn_pooling_params: Optional[AttentionPoolingParams] = None,
        **kwargs,
    ):
        # pretrained_model_name "convnext_tiny.fb_in22k"
        ## forward pass: https://github.com/huggingface/pytorch-image-models/blob/b3e816d6d71ec132b39c603d68b619ae2870fd0a/timm/models/convnext.py#L420

        super().__init__()
        self.cfg = config
        model = create_model(config.pretrained_model_name, pretrained=config.pretrained)  # type: ignore

        self.kernels_per_channel = config.kernels_per_channel
        self.pooling_channel_type = config.pooling_channel_type
        self.enable_sample = config.enable_sample
        self.patch_size = config.patch_size
        self.sample_by_weights = config.sample_by_weights
        self.sample_by_weights_warmup = config.sample_by_weights_warmup
        self.sample_by_weights_scale = config.sample_by_weights_scale

        ## all channels in this order (alphabet): ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']
        # self.mapper = {
        #     "Allen": [5, 2, 6],
        #     "HPA": [3, 6, 5, 0],
        #     "CP": [5, 0, 7, 1, 4],
        # }
        mapper = kwargs["mapper"]
        self.mapper = mapper
        out_dim, original_in_dim, kh, kw = model.stem[0].weight.shape
        self.stride = model.stem[0].stride
        self.padding = model.stem[0].padding
        self.dilation = model.stem[0].dilation
        self.groups = model.stem[0].groups

        total_in_channels = len(config.in_channel_names)

        self.get_patch_emb = nn.ModuleDict()

        ## workaround KeyError: "attribute 'train' already exists" issue of nn.ModuleDict()
        ## https://github.com/pytorch/pytorch/issues/71203
        self.add_prefix = lambda x: f"chunk_{x}"
        if "train" not in self.mapper:  ## CHAMMI?
            for chunk, vals in self.mapper.items():
                self.get_patch_emb[self.add_prefix(chunk)] = nn.Conv2d(
                    len(vals),
                    len(vals),
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                    padding=0,
                    groups=len(vals),
                )
        else:
            self.get_patch_emb = None
        self.kernel_size = 3
        self.conv1depthwise_param_bank = nn.Parameter(
            torch.zeros(total_in_channels * self.kernels_per_channel, 1, self.kernel_size, self.kernel_size)
        )

        if self.pooling_channel_type in [
            ChannelPoolingType.WEIGHTED_SUM_RANDOM,
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_NO_SOFTMAX,
        ]:
            self.weighted_sum_pooling = nn.Parameter(torch.randn(total_in_channels))

        if self.pooling_channel_type in [
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_PAIRWISE_NO_SOFTMAX,
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_PAIRWISE,
        ]:
            self.weighted_sum_pooling = torch.nn.ParameterDict()
            for channel, idxs in self.mapper.items():
                self.weighted_sum_pooling[channel] = nn.Parameter(torch.randn(len(idxs)))

        if self.pooling_channel_type in [
            ChannelPoolingType.WEIGHTED_SUM_ONE,
            ChannelPoolingType.WEIGHTED_SUM_ONE_NO_SOFTMAX,
        ]:
            self.weighted_sum_pooling = nn.Parameter(torch.ones(total_in_channels))

        if self.pooling_channel_type == ChannelPoolingType.ATTENTION:
            self.attn_pooling = FanAttentionV2(4 * 4 * 2 * 64)

        nn.init.kaiming_normal_(self.conv1depthwise_param_bank, mode="fan_in", nonlinearity="relu")

        ## store reference for later access
        # self.adaptive_interface = nn.ParameterList(
        #     [self.get_patch_emb, self.conv_1x1, self.conv1depthwise_param_bank]
        # )
        # if hasattr(self, "weighted_sum_pooling"):
        #     self.adaptive_interface.append(self.weighted_sum_pooling)
        # if hasattr(self, "attn_pooling"):
        #     self.adaptive_interface.append(self.attn_pooling)
        if config.is_conv_small:
            num_stages_1 = 8
            num_stages_2 = 2
        else:
            num_stages_1 = 9
            num_stages_2 = 3

        # self.conv_1x1 = conv1x1(self.kernels_per_channel, out_dim)
        # self.norm = nn.InstanceNorm2d(out_dim, affine=True)

        self.stem = nn.Sequential(
            conv1x1(self.kernels_per_channel, out_dim),
            nn.InstanceNorm2d(out_dim, affine=True),
        )

        ## shared feature_extractor
        self.feature_extractor = nn.Sequential(
            # model.stem[1],
            model.stages[0],
            model.stages[1],
            model.stages[2].downsample,
            *[model.stages[2].blocks[i] for i in range(num_stages_1)],
            model.stages[3].downsample,
            *[model.stages[3].blocks[i] for i in range(num_stages_2)],
        )
        self.classifer_head = nn.Identity()
        if "Allen" not in mapper:  ## if not Morphem dataset
            ## append an classifier layer to the model
            self.classifer_head = nn.Linear(model.num_features, config.num_classes)

        num_proxies = config.num_classes  ## depends on the number of classes of the dataset
        self.dim = 768 if self.cfg.pooling in ["avg", "max", "avgmax"] else 7 * 7 * 768
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = config.temperature  # scale = sqrt(1/T)
        if self.cfg.learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        else:
            self.scale = np.sqrt(1.0 / init_temperature)

    def slice_params_first_layer(
        self,
        chunk: str,
        training_chunks_str: str | None,
        channels: List[int],
        new_channel_init: NewChannelLeaveOneOut | None,
    ) -> Tensor:
        ## form param for the first conv layer
        param_list = []
        ## print w/probablity of 10%
        for c in channels:
            param = self.conv1depthwise_param_bank[
                c * self.kernels_per_channel : (c + 1) * self.kernels_per_channel, ...
            ]
            param_list.append(param)

        ## evaluation with leave one out
        if (not self.training) and (training_chunks_str is not None):
            training_chunks = training_chunks_str.split("_")
            training_channels = [self.mapper[ch] for ch in training_chunks]
            training_channels = [item for sublist in training_channels for item in sublist]  ## flatten
            chs_not_seen = [c for c in training_channels if c not in self.mapper[chunk]]

            ## conv1depthwise_param_bank's shape: (c_total * kernels_per_channel, 1, 3, 3)
            param_list = []
            cur = 0
            avg2, avg2_not_seen = (
                NewChannelLeaveOneOut.AVG_2,
                NewChannelLeaveOneOut.AVG_2_NOT_IN_CHUNK,
            )

            avg3, avg3_not_seen = (
                NewChannelLeaveOneOut.AVG_3,
                NewChannelLeaveOneOut.AVG_3_NOT_IN_CHUNK,
            )
            ch_banks = chs_not_seen if "not_in_chunk" in new_channel_init else training_channels

            cur = 0
            training_chunks = training_chunks_str.split("_")
            training_channels = [self.mapper[ch] for ch in training_chunks]
            # flaten training_channels
            training_channels = [item for sublist in training_channels for item in sublist]
            for c in self.mapper[chunk]:
                if c not in training_channels:
                    if new_channel_init in [avg2, avg2_not_seen]:
                        c1 = ch_banks[cur]
                        c2 = ch_banks[(cur + 1) % len(ch_banks)]
                        param_1 = self.conv1depthwise_param_bank[
                            c1 * self.kernels_per_channel : (c1 + 1) * self.kernels_per_channel, ...
                        ]  # type: ignore
                        param_2 = self.conv1depthwise_param_bank[
                            c2 * self.kernels_per_channel : (c2 + 1) * self.kernels_per_channel, ...
                        ]  # type: ignore
                        param = (param_1 + param_2) / 2
                    elif new_channel_init in [avg3, avg3_not_seen]:
                        c1 = ch_banks[cur]
                        c2 = ch_banks[(cur + 1) % len(ch_banks)]
                        c3 = ch_banks[(cur + 2) % len(ch_banks)]
                        param_1 = self.conv1depthwise_param_bank[
                            c1 * self.kernels_per_channel : (c1 + 1) * self.kernels_per_channel, ...
                        ]  # type: ignore
                        param_2 = self.conv1depthwise_param_bank[
                            c2 * self.kernels_per_channel : (c2 + 1) * self.kernels_per_channel, ...
                        ]  # type: ignore
                        param_3 = self.conv1depthwise_param_bank[
                            c3 * self.kernels_per_channel : (c3 + 1) * self.kernels_per_channel, ...
                        ]  # type: ignore
                        param = (param_1 + param_2 + param_3) / 3
                    elif new_channel_init == NewChannelLeaveOneOut.REPLICATE:
                        c = training_channels[cur]
                        param = self.conv1depthwise_param_bank[
                            c * self.kernels_per_channel : (c + 1) * self.kernels_per_channel, ...
                        ]
                    elif new_channel_init == NewChannelLeaveOneOut.RANDOM:
                        param = self.conv1depthwise_param_bank[
                            c * self.kernels_per_channel : (c + 1) * self.kernels_per_channel, ...
                        ]
                    elif new_channel_init == NewChannelLeaveOneOut.ZERO:
                        param = torch.zeros_like(
                            self.conv1depthwise_param_bank[
                                c * self.kernels_per_channel : (c + 1) * self.kernels_per_channel, ...
                            ]
                        )
                    else:
                        raise ValueError(f"Invalid layer_init: '{new_channel_init}'")
                    cur = (cur + 1) % len(training_channels)
                else:
                    param = self.conv1depthwise_param_bank[
                        c * self.kernels_per_channel : (c + 1) * self.kernels_per_channel, ...
                    ]
                param_list.append(param)

        params = torch.cat(param_list, dim=0)
        return params

    def _reset_params(self, model):
        for m in model.children():
            if len(list(m.children())) > 0:
                self._reset_params(m)

            elif isinstance(m, nn.Conv2d):
                print("resetting", m)
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print("resetting", m)

            elif isinstance(m, nn.Linear):
                print("resetting", m)

                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
            else:
                print("skipped", m)

    def _init_bias(self, model):
        ## Init bias of the first layer
        if model.stem[0].bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(model.stem[0].weight)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(model.stem[0].bias, -bound, bound)

    def forward(
        self,
        x: torch.Tensor,
        chunk_name: str,
        training_chunks: str | None = None,
        init_first_layer=None,  # not used
        new_channel_init: NewChannelLeaveOneOut | None = None,
        **kwargs,
    ) -> torch.Tensor:
        Cin = x.shape[1]

        channels = self.mapper[chunk_name]
        warmup = self.sample_by_weights_warmup
        scale = self.sample_by_weights_scale

        ## if training time, and we use channel sampling
        if self.training and self.enable_sample:
            cur_epoch = kwargs["cur_epoch"]
            Cin_new = random.randint(1, Cin)
            if self.sample_by_weights and cur_epoch > warmup:
                weights = self.weighted_sum_pooling[channels].detach().cpu().numpy()
                weights = scale / np.abs(weights)
                s = np.exp(weights).sum()
                weights = np.exp(weights) / s
                channels = np.random.choice(self.mapper[chunk_name], size=Cin_new, replace=False, p=weights)
            else:
                channels = random.sample(self.mapper[chunk_name], k=Cin_new)
            channels_idx = [self.mapper[chunk_name].index(c) for c in channels]
            Cin = Cin_new
            x = x[:, channels_idx, :, :]
            if self.get_patch_emb is not None:
                patch_weights = self.get_patch_emb[self.add_prefix(chunk_name)].weight[channels_idx]
                x = F.conv2d(x, patch_weights, bias=None, stride=1, groups=Cin)
        else:
            if self.get_patch_emb is not None:
                x = self.get_patch_emb[self.add_prefix(chunk_name)](x)
        if self.get_patch_emb is None and self.patch_size > 1 and self.cfg.reduce_size:
            ## reduce spatial dimension using pooling
            x = F.avg_pool2d(x, self.patch_size, self.patch_size)

        ## slice params of the first layers
        conv1depth_params = self.slice_params_first_layer(
            chunk_name,
            training_chunks_str=training_chunks,
            channels=channels,
            new_channel_init=new_channel_init,
        )

        assert conv1depth_params.shape == (
            Cin * self.kernels_per_channel,
            1,
            self.kernel_size,
            self.kernel_size,
        )
        batch_size = 8  # Adjust this value based on wqavailable memory
        outputs = []
        for i in range(0, x.shape[0], batch_size):
            mini_batch = x[i : i + batch_size]
            out = F.conv2d(mini_batch, conv1depth_params, bias=None, stride=1, padding=1, groups=Cin)
            outputs.append(out)
        out = torch.cat(outputs, dim=0)

        out = rearrange(out, "b (c k) h w -> b c k h w", k=self.kernels_per_channel)
        b, c, k, h, w = out.shape

        if self.pooling_channel_type == ChannelPoolingType.AVG:
            out = out.mean(dim=1)
        elif self.pooling_channel_type == ChannelPoolingType.SUM:
            out = out.sum(dim=1)
        elif self.pooling_channel_type in (
            ChannelPoolingType.WEIGHTED_SUM_RANDOM,
            ChannelPoolingType.WEIGHTED_SUM_ONE,
        ):
            weights = F.softmax(self.weighted_sum_pooling[channels])
            weights = rearrange(weights, "c -> c 1 1 1")
            out = (out * weights).sum(dim=1)
        elif self.pooling_channel_type in (
            ChannelPoolingType.WEIGHTED_SUM_RANDOM_NO_SOFTMAX,
            ChannelPoolingType.WEIGHTED_SUM_ONE_NO_SOFTMAX,
        ):
            weights = self.weighted_sum_pooling[channels]
            weights = rearrange(weights, "c -> c 1 1 1")
            out *= weights
            out = out.sum(dim=1)
        elif self.pooling_channel_type == ChannelPoolingType.WEIGHTED_SUM_RANDOM_PAIRWISE_NO_SOFTMAX:
            weights = self.weighted_sum_pooling[channels]
            weights = rearrange(weights, "c -> c 1 1 1")
            out *= weights
            out = out.sum(dim=1)
        elif self.pooling_channel_type == ChannelPoolingType.ATTENTION:
            out_2 = rearrange(out, "b c k h w -> (b c k) h w")
            x_avg = F.adaptive_avg_pool2d(out_2, (4, 4))
            x_max = F.adaptive_max_pool2d(out_2, (4, 4))
            x_avg = rearrange(x_avg, "(b c k) h w -> b c k (h w)", b=b, k=self.kernels_per_channel)
            x_max = rearrange(x_max, "(b c k) h w -> b c k (h w)", b=b, k=self.kernels_per_channel)
            out_2 = torch.cat([x_avg, x_max], dim=2)
            out_2 = rearrange(out_2, "b c k d -> b c (k d)")
            attn = self.attn_pooling(out_2)  # (b, c)
            weights = rearrange(attn, "b c -> b c 1 1 1")
            out = (out * weights).sum(dim=1)
        else:
            raise ValueError(f"Invalid pooling_channel_type: {self.pooling_channel_type}")
        out = self.stem(out)

        out = self.feature_extractor(out)
        if self.cfg.pooling == FeaturePooling.AVG:
            out = F.adaptive_avg_pool2d(out, (1, 1))
        elif self.cfg.pooling == FeaturePooling.MAX:
            out = F.adaptive_max_pool2d(out, (1, 1))
        elif self.cfg.pooling == FeaturePooling.AVG_MAX:
            x_avg = F.adaptive_avg_pool2d(out, (1, 1))
            x_max = F.adaptive_max_pool2d(out, (1, 1))
            out = torch.cat([x_avg, x_max], dim=1)
        elif self.cfg.pooling == FeaturePooling.NONE:
            pass
        else:
            raise ValueError(f"Pooling {self.cfg.pooling} not supported. Use one of {FeaturePooling.list()}")
        out = rearrange(out, "b c h w -> b (c h w)")
        out = self.classifer_head(out)
        return out


def depthwiseconvnext(cfg: Model, **kwargs) -> DepthwiseConvNeXt:
    return DepthwiseConvNeXt(config=cfg, **kwargs)
