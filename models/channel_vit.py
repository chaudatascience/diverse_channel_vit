### code adapted from: https://github.com/insitro/ChannelViT
import math
import random
from functools import partial
from typing import List, Dict

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
from torch.profiler import profile, record_function, ProfilerActivity

from helper_classes.first_layer_init import NewChannelLeaveOneOut, FirstLayerInit


from utils import trunc_normal_
from models.vit import Block


class PatchEmbedPerChannel(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        mapper: Dict | None = None,
        embed_dim: int = 768,
        enable_sample: bool = True,
        use_channelvit_channels: bool = True,
    ):
        super().__init__()
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
            trunc_normal_(self.channel_embed.weight, std=0.02)

        self.use_channelvit_channels = use_channelvit_channels
        self.enable_sample = enable_sample

    def forward(
        self,
        x,
        chunk_name: str,
        training_chunks,
        new_channel_init: NewChannelLeaveOneOut | None,
        extra_tokens={},
    ):
        # assume all images in the same batch has the same input channels
        cur_channels = self.mapper[chunk_name]  ## type: ignore
        if self.use_channelvit_channels:
            channel_embed = self.channel_embed(tensor(cur_channels, device=x.device))  #  Cin, embed_dim=Cout
            ## repeat the channel embedding for each batch
            channel_embed = repeat(channel_embed, "Cin Cout -> B Cout Cin", B=x.shape[0])

        B, Cin, H, W = x.shape
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
                    else:
                        raise ValueError(f"Invalid new_channel_init: '{new_channel_init}'")
                    cur = (cur + 1) % len(ch_banks)
                else:
                    idx = tensor([c], device=x.device)
                    param = self.channel_embed(idx)
                param_list.append(param)

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
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        enable_sample=False,
        use_channelvit_channels=True,
        **kwargs,
    ):
        super().__init__()

        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.in_chans = in_chans

        self.patch_embed = PatchEmbedPerChannel(
            img_size=img_size[0],
            patch_size=patch_size,
            mapper=mapper,
            in_chans=in_chans,
            embed_dim=embed_dim,
            enable_sample=enable_sample,
            use_channelvit_channels=use_channelvit_channels,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token

        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches // self.in_chans + self.num_extra_tokens, embed_dim)
        )

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
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

        return self.pos_drop(x)

    def forward(
        self,
        x,
        chunk_name: str,
        training_chunks: str | None = None,
        new_channel_init: NewChannelLeaveOneOut | None = None,
        extra_tokens={},
    ):

        x = self.prepare_tokens(x, chunk_name, training_chunks, new_channel_init, extra_tokens)

        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0].clone()

    def get_last_selfattention(self, x, extra_tokens={}):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, extra_tokens={}, n=1):
        x = self.prepare_tokens(x, extra_tokens)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


def channelvit_distill(patch_size=14, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
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


def channelvit_tiny(patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
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


def channelvit_small(patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
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


def channelvit_base(patch_size=16, in_chans=0, mapper=None, **kwargs):
    model = ChannelVisionTransformer(
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


if __name__ == "__main__":
    model = channelvit_tiny()
    print(model)
    ## count params of the model
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(pytorch_total_params / 1e6)
