import numpy as np
from einops import rearrange, reduce
from timm import create_model
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class HyperNetwork(nn.Module):
    def __init__(self, z_dim, d, kernel_size, out_size, in_size=1, use_conv1x1=False):
        super().__init__()
        self.z_dim = z_dim
        self.d = d  ## in the paper, d = z_dim
        self.kernel_size = kernel_size
        self.out_size = out_size
        self.in_size = in_size
        self.use_conv1x1 = use_conv1x1

        self.W = nn.Parameter(torch.randn((self.z_dim, self.in_size, self.d)))
        self.b = nn.Parameter(torch.randn((self.in_size, self.d)))
        if use_conv1x1:
            hid_dim = out_size // 4
            self.W_out_h = nn.Parameter(torch.randn((d, hid_dim, kernel_size, kernel_size)))
            self.W_out = nn.Parameter(torch.randn((hid_dim, out_size)))
        else:
            self.W_out = nn.Parameter(torch.randn((d, out_size, kernel_size, kernel_size)))
        self.b_out = nn.Parameter(torch.randn((out_size, kernel_size, kernel_size)))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.W_out)
        if self.use_conv1x1:
            nn.init.kaiming_normal_(self.W_out_h)

    def forward(self, z: Tensor) -> Tensor:
        """
        @param z: (num_channels, z_dim)
        @return: kernel (out_size, in_size, kernel_size, kernel_size)
        """
        a = torch.einsum("c z, z i d ->c i d", z, self.W) + self.b
        if self.use_conv1x1:
            a = torch.einsum("c i d, d m h w ->c i m h w", a, self.W_out_h)
            a = F.relu(a)
            K = torch.einsum("c i m h w, m o ->c i o h w", a, self.W_out) + self.b_out
        else:
            K = torch.einsum("c i d, d o h w ->c i o h w", a, self.W_out) + self.b_out
        K = rearrange(K, "c i o h w -> o (c i) h w")
        return K


class HyperNetworkV2(nn.Module):
    def __init__(self, z_dim, d, kernel_size, out_size, use_conv1x1=False):
        super().__init__()
        self.z_dim = z_dim
        self.d = d  ## in the paper, d = z_dim
        self.kernel_size = kernel_size
        self.out_size = out_size
        self.use_conv1x1 = use_conv1x1

        self.W = nn.Parameter(torch.randn(self.z_dim, self.d))
        self.b = nn.Parameter(torch.randn(self.d))
        if use_conv1x1:
            hid_dim = out_size // 4
            self.W_out_h = nn.Parameter(torch.randn((d, hid_dim, kernel_size, kernel_size)))
            self.W_out = nn.Parameter(torch.randn((hid_dim, out_size)))
        else:
            self.W_out = nn.Parameter(torch.randn((d, out_size, kernel_size, kernel_size)))
        self.b_out = nn.Parameter(torch.randn((out_size, kernel_size, kernel_size)))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.W_out)
        if self.use_conv1x1:
            nn.init.kaiming_normal_(self.W_out_h)

    def forward(self, z: Tensor) -> Tensor:
        """
        @param z: (num_channels, z_dim)
        @return: kernel (out_size, in_size, kernel_size, kernel_size)
        """
        a = torch.einsum("b c z, z d ->b c d", z, self.W) + self.b
        if self.use_conv1x1:
            a = torch.einsum("b c d, d m h w ->b c m h w", a, self.W_out_h)
            a = F.relu(a)
            K = torch.einsum("b c m h w, m o ->b c o h w", a, self.W_out) + self.b_out
        else:
            K = torch.einsum("b c d, d o h w ->b c o h w", a, self.W_out) + self.b_out
        K = rearrange(K, "b c o h w ->b o c h w")
        return K


class HyperNetworkChannelEmb(nn.Module):
    def __init__(self, z_dim, d, out_size, in_size=1):
        super().__init__()
        self.z_dim = z_dim
        self.d = d  ## in the paper, d = z_dim
        self.out_size = out_size
        self.in_size = in_size

        self.W = nn.Parameter(torch.randn((self.z_dim, self.in_size, self.d)))
        self.b = nn.Parameter(torch.randn((self.in_size, self.d)))
        hid_dim = out_size // 4
        self.W_h = nn.Parameter(torch.randn(d, hid_dim))
        self.W_out = nn.Parameter(torch.randn(hid_dim, out_size))
        self.b_out = nn.Parameter(torch.randn(out_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.W_out)
        nn.init.kaiming_normal_(self.W_h)

    def forward(self, z: Tensor) -> Tensor:
        """
        @param z: (num_channels, z_dim)
        @return: kernel (o, c)
        """
        out = torch.einsum("c z, z i d ->c i d", z, self.W) + self.b
        out = F.relu(out)

        out = torch.einsum("c i d, d h ->c i h", out, self.W_h)
        out = F.relu(out)

        out = torch.einsum("c i h, h o ->c i o", out, self.W_out) + self.b_out
        out = rearrange(out, "c i o -> o c i")
        out = reduce(out, "o c i -> o c", "mean")
        return out


class HyperNetworkChannelEmbV2(nn.Module):
    def __init__(self, z_dim, d, out_size):
        super().__init__()
        self.z_dim = z_dim
        self.d = d  ## in the paper, d = z_dim
        self.out_size = out_size

        self.W = nn.Parameter(torch.randn(self.z_dim, self.d))
        self.b = nn.Parameter(torch.randn(self.d))
        hid_dim = out_size // 4
        self.W_h = nn.Parameter(torch.randn(d, hid_dim))
        self.W_out = nn.Parameter(torch.randn(hid_dim, out_size))
        self.b_out = nn.Parameter(torch.randn(out_size))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.W_out)
        nn.init.kaiming_normal_(self.W_h)

    def forward(self, z: Tensor) -> Tensor:
        """
        @param z: (b, num_channels, z_dim)
        @return: kernel (b o c)
        """
        out = torch.einsum("b c z, z d ->b c d", z, self.W) + self.b
        out = F.relu(out)

        out = torch.einsum("b c d, d h ->b c h", out, self.W_h)
        out = F.relu(out)

        out = torch.einsum("b c h, h o ->b c o", out, self.W_out) + self.b_out
        out = rearrange(out, "b c o ->b o c")
        return out
