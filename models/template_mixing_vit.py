import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
import random
import math
from functools import partial
import torch.nn.functional as F
from torch import Tensor
import torch
import warnings
import os
from typing import Tuple
from einops import rearrange, repeat
from config import Model

from utils import trunc_normal_


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tuple[Tensor, None]:
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class MemEffAttention_v2(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tuple[Tensor, None]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.reshape([B, N, C])
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, None


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        num_gpus = torch.cuda.device_count()
        AttentionClass = MemEffAttention if XFORMERS_AVAILABLE and num_gpus > 0 else Attention
        self.attn = AttentionClass(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if (
            return_attention
        ):  ## TODO: check if this is using efficient attention, which won't return attention weights
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class TemplateMixingViT(nn.Module):
    def __init__(
        self,
        config,
        img_size=[224],
        patch_size=16,
        in_chans=3,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        input_drop=0.0,
        kernel_size=3,
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = self.out_dim = embed_dim
        self.cfg = config
        self.enable_sample = config.enable_sample
        self.patch_size = patch_size

        mapper = kwargs.get("mapper", None)
        self.mapper = mapper

        num_patches = (img_size[0] // patch_size) * (img_size[-1] // patch_size)

        num_templates = len(config.in_channel_names) * config.num_templates_per_channel
        ## all channels in this order (alphabet): config.in_channel_names = ['er', 'golgi', 'membrane', 'microtubules','mito','nucleus','protein', 'rna']        self.enable_sample = self.cfg.enable_sample

        num_proxies = config.num_classes  ## depends on the number of classes of the dataset
        self.dim = 384 if self.cfg.pooling in ["avg", "max", "avgmax"] else 7 * 7 * 768
        self.proxies = torch.nn.Parameter((torch.randn(num_proxies, self.dim) / 8))
        init_temperature = config.temperature  # scale = sqrt(1/T)
        if self.cfg.learnable_temp:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / init_temperature))
        else:
            self.scale = np.sqrt(1.0 / init_temperature)

        # First conv layer
        kernel_size = 3
        hdim_out = self.embed_dim // 8

        self.conv1_param_bank = nn.Parameter(torch.zeros(hdim_out, num_templates, kernel_size, kernel_size))
        self.conv1_coefs = nn.Parameter(torch.zeros(len(config.in_channel_names), num_templates))
        self.conv1x1 = nn.Conv2d(hdim_out, embed_dim, kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.conv1_param_bank, mode="fan_in", nonlinearity="relu")
        nn.init.orthogonal_(self.conv1_coefs)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.num_extra_tokens = 1  # cls token

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.input_drop = nn.Dropout2d(p=input_drop, inplace=True)
        # self.drop_prob = torch.ones(in_chans) * input_drop

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
        self.classifer_head = None
        if "Allen" not in mapper:  ## if not Morphem dataset
            ## append an classifier layer to the model
            self.classifer_head = nn.Linear(embed_dim, config.num_classes)

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

    def interpolate_pos_encoding(self, x, w, h):
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
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def mix_templates_first_layer(self, chunk) -> Tensor:
        """
        @return: return a tensor, shape (out_channels, in_channels, kernel_h, kernel_w)
        """
        coefs = self.conv1_coefs[self.mapper[chunk]]

        coefs = rearrange(coefs, "c t ->1 c t 1 1")
        templates = repeat(self.conv1_param_bank, "o t h w -> o c t h w", c=len(self.mapper[chunk]))
        params = torch.sum(coefs * templates, dim=2)
        return params

    def prepare_tokens(
        self,
        x,
        chunk_name: str,
        training_chunks: str | None = None,
        init_first_layer=None,  # not used
        new_channel_init=None,
        **kwargs,
    ):
        B, Cin, h, w = x.shape

        x = self.input_drop(x)
        cur_channels = self.mapper[chunk_name]  ## type: ignore
        conv1_params = self.mix_templates_first_layer(chunk_name)
        if self.training and self.enable_sample:
            Cin_new = random.randint(1, Cin)
            cur_channels = random.sample(cur_channels, k=Cin_new)
            Cin = Cin_new
            channels_idx = [self.mapper[chunk_name].index(c) for c in cur_channels]
            x = x[:, channels_idx, :, :]
            conv1_params = conv1_params[:, channels_idx, :, :]
            # print("x.shape", x.shape)

        x = F.conv2d(x, conv1_params, bias=None, stride=self.patch_size, padding=0)

        x = self.conv1x1(x)
        x = x.flatten(2).transpose(1, 2)

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.interpolate_pos_encoding(x, w, h)
        return self.pos_drop(x)

    def forward(
        self,
        x,
        chunk_name: str,
        training_chunks: str | None = None,
        init_first_layer=None,  # not used
        new_channel_init=None,
        extra_tokens={},
        **kwargs,
    ):
        x = self.prepare_tokens(x, chunk_name, training_chunks, init_first_layer, new_channel_init)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        out = x[:, 0].clone()
        if self.classifer_head is not None:
            out = self.classifer_head(out)
        return out


def template_mixing_vit_small(cfg, **kwargs):
    model = TemplateMixingViT(
        config=cfg,
        img_size=cfg.img_size,
        patch_size=cfg.patch_size,
        embed_dim=384,
        depth=12,
        in_chans=len(cfg.in_channel_names),
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def templatemixingvit(cfg: Model, **kwargs) -> TemplateMixingViT:
    return template_mixing_vit_small(cfg, **kwargs)
