import numpy as np
import torch
from torch import nn
import sys
from einops import rearrange
from config import Model
from collections import Counter
import torch.nn.functional as F

# from models.channel_vit import channelvit_distill, channelvit_tiny, channelvit_base, channelvit_small
from helper_classes.first_layer_init import NewChannelLeaveOneOut

### code adapted from: https://github.com/insitro/ChannelViT
import math
import random
from functools import partial
from typing import List, Dict, Optional

from collections import defaultdict
import torch
from einops import repeat
from torch import tensor
import torch.distributed as dist
import torch.nn as nn
import os
import sys
from utils import get_gpu_mem

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# export PYTHONPATH=$(pwd)
# from torch.profiler import profile, record_function, ProfilerActivity
from helper_classes.first_layer_init import NewChannelLeaveOneOut, FirstLayerInit


from utils import trunc_normal_
from models.vit import Block, BlockV2


class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        config,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        mapper: Dict | None = None,
        embed_dim: int = 768,
        enable_sample: bool = True,
        use_channelvit_channels: bool = True,
    ):
        super().__init__()
        self.cfg = config
        num_patches = (img_size // patch_size) * (img_size // patch_size) * in_chans
        self.img_size = img_size
        self.mapper = mapper
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv3d(
            1,
            embed_dim,
            kernel_size=(1, patch_size, patch_size),
            stride=(1, patch_size, patch_size),
        )  # CHANGED
        if use_channelvit_channels:
            self.channel_embed = nn.Embedding(in_chans, embed_dim)
            if self.cfg.orthogonal_channel_emb_init:
                print("---------------- use_channelvit_channels=True")
                nn.init.orthogonal_(self.channel_embed.weight)
            else:
                trunc_normal_(self.channel_embed.weight, std=0.02)

            if self.cfg.freeze_channel_emb:
                self.channel_embed.weight.requires_grad = False
                print("---------- Froze channel embedding!")

        self.use_channelvit_channels = use_channelvit_channels
        self.enable_sample = enable_sample

    def forward(
        self,
        x,
        chunk_name: str,
        training_chunks,
        new_channel_init: NewChannelLeaveOneOut | None,
        extra_tokens={},
        **kwargs,
    ):

        # assume all images in the same batch has the same input channels
        cur_channels = self.mapper[chunk_name]  ## type: ignore
        if self.use_channelvit_channels:
            channel_embed = self.channel_embed(tensor(cur_channels, device=x.device))  #  Cin, embed_dim=Cout
            ## repeat the channel embedding for each batch
            channel_embed = repeat(channel_embed, "Cin Cout -> B Cout Cin", B=x.shape[0])

        b, Cin, h, w = x.shape
        # Note: The current number of channels (Cin) can be smaller or equal to in_chans
        ## if training time, and we use channel sampling
        if self.training and self.enable_sample:
            Cin_new = random.randint(1, Cin)
            cur_channels = random.sample(cur_channels, k=Cin_new)
            Cin = Cin_new
            channels_idx = [self.mapper[chunk_name].index(c) for c in cur_channels]
            x = x[:, channels_idx, :, :]
            if self.use_channelvit_channels:
                channel_embed = channel_embed[:, :, channels_idx]

        ## if test time, and it's leave one out: create emb for novel channels
        if self.use_channelvit_channels and (not self.training) and (training_chunks is not None):
            training_chunks = training_chunks.split("_")
            training_channels = [self.mapper[ch] for ch in training_chunks]
            training_channels = [item for sublist in training_channels for item in sublist]  ## flatten
            chs_not_seen = [c for c in training_channels if c not in self.mapper[chunk_name]]

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

            for c in self.mapper[chunk_name]:
                if c not in training_channels:
                    if new_channel_init in [avg2, avg2_not_seen]:
                        c1 = ch_banks[cur]
                        c2 = ch_banks[(cur + 1) % len(ch_banks)]
                        idx = tensor([c1, c2], device=x.device)
                        param = self.channel_embed(idx).mean(dim=0, keepdim=True)
                    elif new_channel_init in [avg3, avg3_not_seen]:
                        c1 = ch_banks[cur]
                        c2 = ch_banks[(cur + 1) % len(ch_banks)]
                        c3 = ch_banks[(cur + 2) % len(ch_banks)]
                        idx = tensor([c1, c2, c3], device=x.device)
                        param = self.channel_embed(idx).mean(dim=0, keepdim=True)
                    elif new_channel_init == NewChannelLeaveOneOut.REPLICATE:
                        c = ch_banks[cur]
                        idx = tensor([c], device=x.device)
                        param = self.channel_embed(idx)
                    elif new_channel_init == NewChannelLeaveOneOut.ZERO:
                        param = torch.zeros_like(self.channel_embed.weight[0]).unsqueeze(0)
                    elif new_channel_init == NewChannelLeaveOneOut.RANDOM:
                        param = self.channel_embed.weight[c].unsqueeze(0)
                    elif new_channel_init == NewChannelLeaveOneOut.DYNAMIC_INPUT_CORR_1:
                        if not hasattr(self, "bank"):
                            raise ValueError("provide a channel_map (dict)!")
                        ch_idx = self.mapper[chunk_name].index(c)
                        xc_test = x[:, ch_idx].reshape(b, -1)  # (b, h * w)
                        bank = rearrange(self.bank.to(x.device), "a c h w -> a c (h w)")
                        xc_test = F.normalize(xc_test, p=2, dim=-1)
                        bank = F.normalize(bank, p=2, dim=-1)
                        corr = torch.einsum("b d, a c d->b a c", xc_test, bank)
                        corr = torch.argmax(corr, dim=-1)  ## shape b a
                        train_inds = torch.mode(corr, dim=1).values  # shape (b)
                        replicate_channels = [training_channels[idx] for idx in train_inds]
                        param = self.channel_embed(
                            tensor(replicate_channels, device=x.device)
                        )  ## shape (b, embed_dim)
                        # print(ch_idx, c)
                        # print(Counter(train_inds.detach().cpu().numpy()))
                    elif new_channel_init == NewChannelLeaveOneOut.DYNAMIC_INPUT_CORR_2:
                        if not hasattr(self, "bank"):
                            raise ValueError("provide a channel_map (dict)!")
                        ch_idx = self.mapper[chunk_name].index(c)
                        xc_test = x[:, ch_idx].reshape(b, -1)  # (b, h * w)
                        xc_test = repeat(xc_test, "b d -> b (a d)", a=self.bank.shape[0])
                        bank = rearrange(self.bank.to(x.device), "a c h w -> c (a h w)")
                        xc_test = F.normalize(xc_test, p=2, dim=-1)
                        bank = F.normalize(bank, p=2, dim=-1)
                        corr = torch.einsum("b h, c h ->b c", xc_test, bank)
                        train_inds = torch.argmax(corr, dim=-1)  # shape (b)
                        replicate_channels = [training_channels[idx] for idx in train_inds]
                        param = self.channel_embed(
                            tensor(replicate_channels, device=x.device)
                        )  ## shape (b, embed_dim)
                    elif new_channel_init == NewChannelLeaveOneOut.DYNAMIC_INPUT_CORR_3:
                        if not hasattr(self, "bank"):
                            raise ValueError("provide a channel_map (dict)!")
                        ch_idx = self.mapper[chunk_name].index(c)
                        xc_test = x[:, ch_idx].reshape(b, -1)  # (b, h * w)
                        bank = rearrange(self.bank.to(x.device), "a c h w -> a c (h w)")
                        bank = bank.mean(dim=0)  # (c, h * w)
                        xc_test = F.normalize(xc_test, p=2, dim=-1)
                        bank = F.normalize(bank, p=2, dim=-1)
                        corr = torch.einsum("b d, c d->b c", xc_test, bank)
                        train_inds = torch.argmax(corr, dim=-1)  # shape (b)
                        replicate_channels = [training_channels[idx] for idx in train_inds]
                        param = self.channel_embed(
                            tensor(replicate_channels, device=x.device)
                        )  ## shape (b, embed_dim)
                    elif new_channel_init == NewChannelLeaveOneOut.DYNAMIC_INPUT_CORR_4:
                        if not hasattr(self, "bank"):
                            raise ValueError("provide a channel_map (dict)!")
                        ch_idx = self.mapper[chunk_name].index(c)
                        xc_test = x[:, ch_idx].reshape(b, -1)  # (b, h * w)
                        bank = rearrange(self.bank.to(x.device), "a c h w -> a c (h w)")
                        xc_test = F.normalize(xc_test, p=2, dim=-1)
                        bank = F.normalize(bank, p=2, dim=-1)
                        corr = torch.einsum("b d, a c d->b a c", xc_test, bank)
                        corr = torch.mean(corr, dim=1)  ## shape b c
                        train_embs = self.channel_embed(torch.tensor(training_channels, device="cuda"))
                        param = torch.einsum("b c, c d -> b d", corr, train_embs)  ## shape (b, embed_dim)
                    elif new_channel_init == NewChannelLeaveOneOut.DYNAMIC_INPUT_CORR_5:
                        if not hasattr(self, "bank"):
                            raise ValueError("provide a channel_map (dict)!")
                        ch_idx = self.mapper[chunk_name].index(c)
                        xc_test = x[:, [ch_idx]]  # (b, 1, h, w)
                        bank = self.bank.to(x.device)  # (a c h w)
                        xc_test = self.proj(xc_test.unsqueeze(1))  # (b, embed_dim, h, w)
                        xc_test = rearrange(xc_test, "b d 1 h w -> b (1 d h w)")
                        bank = self.proj(bank.unsqueeze(1))  # (a c d h w)
                        bank = rearrange(bank, "a d c h w -> a c (d h w)")
                        xc_test = F.normalize(xc_test, p=2, dim=-1)
                        bank = F.normalize(bank, p=2, dim=-1)
                        corr = torch.einsum("b d, a c d->b a c", xc_test, bank)
                        corr = torch.mean(corr, dim=1)  ## shape b c
                        train_embs = self.channel_embed(torch.tensor(training_channels, device="cuda"))
                        param = torch.einsum("b c, c d -> b d", corr, train_embs)  ## shape (b, embed_dim)
                    elif new_channel_init == NewChannelLeaveOneOut.DYNAMIC_INPUT_CORR_6:
                        if not hasattr(self, "bank"):
                            raise ValueError("provide a channel_map (dict)!")
                        ch_idx = self.mapper[chunk_name].index(c)
                        xc_test = x[:, [ch_idx]]  # (b, 1, h, w)
                        bank = self.bank.to(x.device)  # (a c h w)
                        xc_test = self.proj(xc_test.unsqueeze(1))  # (b, embed_dim, h, w)
                        xc_test = rearrange(xc_test, "b d 1 h w -> b (1 d h w)")
                        bank = self.proj(bank.unsqueeze(1))  # (a c d h w)
                        bank = rearrange(bank, "a d c h w -> a c (d h w)")
                        xc_test = F.normalize(xc_test, p=2, dim=-1)
                        bank = F.normalize(bank, p=2, dim=-1)
                        corr = torch.einsum("b d, a c d->b a c", xc_test, bank)
                        corr = torch.argmax(corr, dim=-1)  ## shape b a
                        train_inds = torch.mode(corr, dim=1).values  # shape (b)
                        replicate_channels = [training_channels[idx] for idx in train_inds]
                        param = self.channel_embed(
                            tensor(replicate_channels, device=x.device)
                        )  ## shape (b, embed_dim)
                    elif new_channel_init == NewChannelLeaveOneOut.FIXED_INPUT_CORR:
                        if not hasattr(self, "channel_map"):
                            raise ValueError("provide a channel_map (dict)!")
                        param = self.channel_embed(tensor([self.channel_map[c]], device=x.device))
                    elif new_channel_init == NewChannelLeaveOneOut.RANDOM_INPUT_CORR:
                        c_new = np.random.choice(training_channels)
                        # print(f"{c} -> {c_new}")
                        param = self.channel_embed(tensor([c_new], device=x.device))
                    else:
                        raise ValueError(f"Invalid new_channel_init: '{new_channel_init}'")
                    cur = (cur + 1) % len(ch_banks)
                else:
                    idx = tensor([c], device=x.device)
                    param = self.channel_embed(idx)
                    if "dynamic_input_corr" in new_channel_init:
                        param = repeat(param, "1 emb -> b emb", b=b)
                param_list.append(param)

            if "dynamic_input_corr" in new_channel_init:
                channel_embed = torch.stack(param_list, dim=-1)  # B Cout Cin
            else:
                channel_embed = torch.cat(param_list, dim=0)
                channel_embed = repeat(channel_embed, "Cin Cout -> B Cout Cin", B=x.shape[0])

        # shared projection layer across channels
        x = self.proj(x.unsqueeze(1))  # B Cout Cin H W

        # channel specific offsets
        if self.use_channelvit_channels:
            x += channel_embed.unsqueeze(-1).unsqueeze(-1)

        # preparing the output sequence
        x = x.flatten(2)  # B Cout CinHW
        x = x.transpose(1, 2)  # B CinHW Cout

        return x, Cin


class ChannelVisionTransformer(nn.Module):
    """Channel Vision Transformer"""

    def __init__(
        self,
        config,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        mapper: Dict | None = None,
        num_classes=0,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        # drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_sample=False,
        use_channelvit_channels=True,
        **kwargs,
    ):
        super().__init__()
        self.cfg = config
        drop_path_rate = config.drop_path_rate
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans

        self.counter = defaultdict(lambda: 0)

        self.patch_embed = PatchEmbedPerChannel(
            config=config,
            img_size=img_size[0],
            patch_size=patch_size,
            mapper=mapper,
            in_chans=in_chans,
            embed_dim=embed_dim,
            enable_sample=enable_sample,
            use_channelvit_channels=use_channelvit_channels,
        )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        print("----dpr", dpr)

        if self.cfg.block_type == "block_v2":
            BlockClass = BlockV2
        elif self.cfg.block_type == "block":
            BlockClass = Block
        else:
            raise ValueError(f"Unknown block type: {self.cfg.block_type}")

        self.blocks = nn.ModuleList(
            [
                BlockClass(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    **kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h, nc):
        # number of auxilary dimensions before the patches
        if not hasattr(self, "num_extra_tokens"):
            # backward compatibility
            num_extra_tokens = 1
        else:
            num_extra_tokens = self.num_extra_tokens

        npatch = x.shape[1] - num_extra_tokens
        N = self.pos_embed.shape[1] - num_extra_tokens

        if npatch == N and w == h:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, :num_extra_tokens]
        patch_pos_embed = self.pos_embed[:, num_extra_tokens:]

        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, 1, -1, dim)

        # create copies of the positional embeddings for each channel
        patch_pos_embed = patch_pos_embed.expand(1, nc, -1, dim).reshape(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def prepare_tokens(self, x, chunk: str, training_chunks_str, new_channel_init, extra_tokens):
        B, _, w, h = x.shape
        x, nc = self.patch_embed(
            x, chunk, training_chunks_str, new_channel_init, extra_tokens
        )  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h, nc)

        ### drop some tokens randomly at the last dim
        if self.cfg.dropout_tokens_hcs == "random" and self.training:  ## x: [B CinHW Cout]
            cinHW = x.shape[1]
            HW = cinHW // nc
            cinHW_new = random.randint(1, nc) * HW
            drops_true = random.sample(range(cinHW), k=cinHW_new)
            ## make sure the first token ([CLS]) is not dropped
            drops = [True]
            for i in range(1, cinHW):
                if i in drops_true:
                    drops.append(True)
                else:
                    drops.append(False)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]
        elif self.cfg.dropout_tokens_hcs == "channel" and self.training:
            cinHW = x.shape[1]
            HW = cinHW // nc
            cin_new = random.randint(1, nc)
            ## choose cin_new from nc channels
            drops_channels = random.sample(range(nc), k=cin_new)
            drops = [True]  ## make sure the first token ([CLS]) is not dropped
            for i in range(nc):
                if i in drops_channels:
                    tmp = [True] * HW
                else:
                    tmp = [False] * HW
                drops.extend(tmp)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]

        elif self.cfg.dropout_tokens_hcs == "channel_random50" and self.training:
            cinHW = x.shape[1]
            HW = cinHW // nc
            ## get ceil(50% of the channels)
            cin_new = int(math.ceil(0.5 * nc))
            ## choose cin_new from nc channels
            drops_channels = random.sample(range(nc), k=cin_new)
            drops = [True]  ## make sure the first token ([CLS]) is not dropped
            for i in range(nc):
                if i in drops_channels:
                    tmp = [True] * HW
                else:
                    tmp = [False] * HW
                drops.extend(tmp)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]
        if self.cfg.dropout_tokens_hcs == "token_random50" and self.training:  ## x: [B CinHW Cout]
            cinHW = x.shape[1]
            HW = cinHW // nc
            cinHW_new = int(math.ceil(0.5 * nc)) * HW
            drops_true = random.sample(range(cinHW), k=cinHW_new)
            ## make sure the first token ([CLS]) is not dropped
            drops = [True]
            for i in range(1, cinHW):
                if i in drops_true:
                    drops.append(True)
                else:
                    drops.append(False)
            drops = torch.tensor(drops, device=x.device)
            x = x[:, drops, :]

        return self.pos_drop(x)

    def forward(
        self,
        x,
        chunk_name: str,
        training_chunks: str | None = None,
        new_channel_init: NewChannelLeaveOneOut | None = None,
        extra_tokens={},
    ):
        B, _, w, h = x.shape
        x = self.prepare_tokens(x, chunk_name, training_chunks, new_channel_init, extra_tokens)
        nc = x.shape[1] // ((w // self.patch_size) * (h // self.patch_size))

        for blk in self.blocks:
            if isinstance(blk, BlockV2):
                x, counter = blk(x, pruning_method=self.cfg.dropout_tokens_hcs, nc=nc)
                for k, v in counter.items():
                    self.counter[k] += v
            else:
                x = blk(x)

        x = self.norm(x)
        return x[:, 0].clone()

    def get_last_selfattention(self, x, extra_tokens={}, chunk="", layer_idx=-1):
        x = self.prepare_tokens(
            x, chunk=chunk, training_chunks_str=None, new_channel_init=None, extra_tokens=extra_tokens
        )
        print(f"------- there are {len(self.blocks)} blocks")
        for i, blk in enumerate(self.blocks):
            if i == layer_idx:
                return blk(x, return_attention=True)

            x = blk(x)

    def get_intermediate_layers(self, x, extra_tokens={}, n=1):
        x = self.prepare_tokens(x, extra_tokens)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def channelvit_distill(config, patch_size=14, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        mapper=mapper,
        in_chans=in_chans,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_tiny(config, patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        in_chans=in_chans,
        mapper=mapper,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_small(config, patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        in_chans=in_chans,
        mapper=mapper,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def channelvit_base(config, patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
        config=config,
        img_size=config.img_size,
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        in_chans=in_chans,
        mapper=mapper,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


class ChannelViTAdapt(nn.Module):
    def __init__(self, config: Model, **kwargs):
        super().__init__()
        self.cfg = config

        ## TODO: should pass this in `forward()``
        # mapper = {
        #     "Allen": [5, 2, 6],
        #     "HPA": [3, 6, 5, 0],
        #     "CP": [5, 0, 7, 1, 4],
        # }

        mapper = kwargs["mapper"]

        total_in_channels = len(config.in_channel_names)

        ## channelvit_distill 21,470,208 params
        if config.pretrained_model_name == "distill":
            model = channelvit_distill(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        elif config.pretrained_model_name == "tiny":
            model = channelvit_tiny(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        elif config.pretrained_model_name == "base":
            model = channelvit_base(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        elif config.pretrained_model_name == "small":
            model = channelvit_small(
                config=config,
                patch_size=config.patch_size,
                in_chans=total_in_channels,
                mapper=mapper,
                enable_sample=config.enable_sample,
                use_channelvit_channels=config.use_channelvit_channels,
            )
        else:
            raise ValueError("Unknown model name")

        ## TODO: add options to freeze some layers
        self.feature_extractor = model
        self.classifer_head = nn.Identity()

        if "Allen" not in mapper:  ## if not Morphem dataset
            ## append an classifier layer to the model
            self.classifer_head = nn.Linear(model.num_features, config.num_classes)

        num_proxies = config.num_classes  ## depends on the number of classes of the dataset
        self.dim = model.norm.weight.shape[0]
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = config.temperature  # scale = sqrt(1/T)
        if self.cfg.learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        else:
            self.scale = np.sqrt(1.0 / init_temperature)

        self.adaptive_interface = nn.ParameterList([self.proxies])
        # 21477504

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
        training_chunks: Optional[str] = None,
        init_first_layer=None,
        new_channel_init: Optional[NewChannelLeaveOneOut] = None,
        **kwargs,
    ) -> torch.Tensor:
        # init_first_layer: not used
        x = self.feature_extractor(x, chunk_name, training_chunks, new_channel_init)
        # if not self.training:
        #     ## save to file
        #     idx = kwargs.get("idx", "")
        #     name = f"{chunk_name}_{idx}.pt"
        #     torch.save(x, name)
        #     print("x saved to", name)
        x = self.classifer_head(x)
        return x


def channelvit_adapt(cfg: Model, **kwargs) -> ChannelViTAdapt:
    return ChannelViTAdapt(config=cfg, **kwargs)
