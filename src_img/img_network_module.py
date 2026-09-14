import torch.nn as nn


"""Discriminator"""


def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        ksize=3,
        pad=1,
        d_spectral_norm=False,
        activation=nn.ReLU(),
    ):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=ksize, padding=pad)
        self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class DisBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels=None,
        ksize=3,
        pad=1,
        d_spectral_norm=False,
        activation=nn.ReLU(),
        downsample=False,
    ):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self.c1 = nn.Conv2d(
            in_channels, hidden_channels, kernel_size=ksize, padding=pad
        )
        self.c2 = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=ksize, padding=pad
        )
        if d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)


class Discriminator(nn.Module):
    def __init__(
        self, df_dim, in_channel=3, d_spectral_norm=False, activation=nn.PReLU()
    ):
        super(Discriminator, self).__init__()
        self.ch = df_dim
        self.activation = activation

        # print(type(df_dim), df_dim)
        # print(type(in_channel), in_channel)

        self.block1 = OptimizedDisBlock(
            in_channel, self.ch, d_spectral_norm=d_spectral_norm
        )
        self.block2 = DisBlock(
            self.ch,
            self.ch,
            d_spectral_norm=d_spectral_norm,
            activation=activation,
            downsample=True,
        )
        self.block3 = DisBlock(
            self.ch,
            self.ch,
            d_spectral_norm=d_spectral_norm,
            activation=activation,
            downsample=False,
        )
        self.block4 = DisBlock(
            self.ch,
            self.ch,
            d_spectral_norm=d_spectral_norm,
            activation=activation,
            downsample=False,
        )
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        h = self.block1(h)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output
    

    # def forward(self, x):
    #     h = x

    #     h = self.block1(h)
    #     print("after block1:", h.shape)

    #     h = self.block2(h)
    #     print("after block2:", h.shape)

    #     h = self.block3(h)
    #     print("after block3:", h.shape)

    #     h = self.block4(h)
    #     print("after block4:", h.shape)

    #     h = self.activation(h)
    #     print("after activation:", h.shape)

    #     h = h.sum(2).sum(2)
    #     print("after pooling:", h.shape)

    #     output = self.l5(h)
    #     print("after linear:", output.shape)

    #     return output
    


# pylint: skip-file
import math
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# helpers functions

__all__ = ["Unet"]


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose2d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1) + 1e-5)

    def forward(self, x):
        return self.fn(x) * self.g


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1), nn.GroupNorm(groups, dim_out), Mish()
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(
            qkv, "b (qkv heads c) h w -> qkv b heads c (h w)", heads=self.heads, qkv=3
        )
        k = k.softmax(dim=-1)
        context = torch.einsum("bhdn,bhen->bhde", k, v)
        out = torch.einsum("bhde,bhdn->bhen", context, q)
        out = rearrange(
            out, "b heads c (h w) -> b (heads c) h w", heads=self.heads, h=h, w=w
        )
        return self.to_out(out)


# model


class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), in_channel=3):
        super().__init__()
        dims = [in_channel, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out),
                        ResnetBlock(dim_out, dim_out),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in),
                        ResnetBlock(dim_in, dim_in),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, in_channel)
        self.final_conv = nn.Sequential(
            Block(dim, dim), nn.Conv2d(dim, out_dim, 1), nn.Tanh()
        )

    def forward(self, x):
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x)
            x = resnet2(x)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x)
            x = resnet2(x)
            x = attn(x)
            x = upsample(x)

        return self.final_conv(x)
    

def _weight_initialization_function(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
        
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.00, 0.02)
            nn.init.zeros_(m.bias)
        
        elif isinstance(m, nn.GroupNorm):
            nn.init.normal_(m.weight, 1.00, 0.02)
            nn.init.zeros_(m.bias)


if __name__ == "__main__":
    from torchinfo import summary
    import copy

    unet = Unet(dim=32, in_channel=1)
    summary(unet, input_size=(7, 1, 32, 32))

    dual_net = Discriminator(df_dim=128, in_channel=1, d_spectral_norm=True, activation=nn.PReLU()).to("cuda")
    dual_net.apply(_weight_initialization_function)
    last_dual_net = copy.deepcopy(dual_net).eval()


    