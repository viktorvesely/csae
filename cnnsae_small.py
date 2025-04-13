import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def top_r_channels(x: torch.Tensor, r: int) -> torch.Tensor:
    """Selects olny top-r CNN channels from the x activations
    """

    if r <= 0 or r >= x.size(1):
        return x

    b, c, h, w = x.shape
    x_2d = x.permute(0, 2, 3, 1).reshape(b * h * w, c)

    top_vals, top_idx = torch.topk(x_2d, k=r, dim=1)

    mask = torch.zeros_like(x_2d, dtype=torch.bool)
    mask.scatter_(1, top_idx, True)

    x_pruned = x_2d * mask
    x_pruned = x_pruned.view(b, h, w, c).permute(0, 3, 1, 2)

    return x_pruned


class GroupedFullyConnected(nn.Module):

    def __init__(
        self,
        channels: int,
        groups: int,
        w: int = 8,
        h: int = 8,
        ):

        super().__init__()

        assert (channels % groups) == 0

        self.channels = channels
        self.w = w
        self.h = h
        self.groups = groups
        self.group_shape = (channels, w, h)

        channels_per_group = channels // groups
        weights_per_group = channels_per_group * w * h

        self.channels_per_group = channels_per_group

        linears = []
        for i_group in range(groups):
            linears.append(nn.Sequential(
                nn.Flatten(),
                nn.Linear(weights_per_group, weights_per_group),
                nn.Unflatten(1, (channels_per_group, w, h))
            ))

        self.linears = nn.ModuleList(linears)

    def forward(self, x):

        y = []
        for i_group in range(self.groups):
            start = i_group * self.channels_per_group
            end = start + self.channels_per_group
            group_x = x[:, start:end, :, :]
            group_y = self.linears[i_group](group_x)
            y.append(group_y)

        y = torch.cat(y, dim=1)
        return y



class RandomGroupedFullyConnected(GroupedFullyConnected):

    def __init__(self, channels, groups, w = 8, h = 8):
        super().__init__(channels, groups, w, h)

        self.permute_channels_inds = torch.randperm(channels)

    def forward(self, x):
        x = x[:, self.permute_channels_inds, :, :]
        return super().forward(x)


class CSAE(nn.Module):
    def __init__(
        self,
        in_channels=256,
        ls_factor=5,
        kernel_size=3,
        pooling_size=2,
        r_values=(10,),  # top-r competition per layer
        sparsity_lambda: float = 1.0,
        contrastive_lambda: float = 1.0,
        GFC_divisor: int = 1,
        spatial: int = 8
    ):

        print("lsfactor" , ls_factor)
        print("topr", r_values)

        super(CSAE, self).__init__()

        self.sparsity_lambda = sparsity_lambda
        self.contrastive_lambda = contrastive_lambda

        channels = [in_channels * i for i in range(1, ls_factor + 1)]

        pre_convs = []
        for i in range(len(channels) - 1):

            # Groupped fully connected
            pre_convs.append(
                GroupedFullyConnected(
                    channels=channels[i],
                    groups=in_channels // GFC_divisor
                )
            )
            pre_convs.append(nn.LeakyReLU())

            # Cross channel
            pre_convs.append(nn.Conv2d(
                channels[i],
                channels[i],
                kernel_size=1,
                padding=0
            ))
            pre_convs.append(nn.LeakyReLU())

            # Cross spatial
            pre_convs.append(nn.Conv2d(
                channels[i],
                channels[i + 1],
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=in_channels))
            pre_convs.append(nn.LeakyReLU())

        self.pre_convs = nn.Sequential(
            *pre_convs
        )

        # ----- Encoder Convolutions -----
        self.encoder_conv1 = nn.Conv2d(
            channels[-1],
            channels[-1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels
        )

        self.pool1 = nn.MaxPool2d(pooling_size, stride=2, return_indices=True)
        self.pool1_rev = nn.MaxUnpool2d(pooling_size, stride=2)

        self.decoder_conv2 = nn.Conv2d(
            channels[-1],
            channels[-1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels
        )

        post_convs = []
        rev_channels = list(reversed(channels))
        for i in range(len(rev_channels) - 1):

            # Cross spatial
            post_convs.append(
                nn.Conv2d(
                    rev_channels[i],
                    rev_channels[i + 1],
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                    groups=in_channels
                ))
            post_convs.append(nn.LeakyReLU())

            # Cross channel
            post_convs.append(nn.Conv2d(
                rev_channels[i + 1],
                rev_channels[i + 1],
                kernel_size=1,
                padding=0
            ))
            post_convs.append(nn.LeakyReLU())

            # Groupped fully connected
            post_convs.append(
                GroupedFullyConnected(
                    channels=rev_channels[i + 1],
                    groups=in_channels // GFC_divisor
                )
            )
            if i != len(rev_channels) - 2:
                post_convs.append(nn.LeakyReLU())

        self.post_convs = nn.Sequential(
            *post_convs
        )

        self.in_channels = in_channels
        self.ls_factor = ls_factor
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.stride = 2
        self.r_values = r_values
        self.spatial_dim = spatial
        self.decoded_numel = spatial * spatial * in_channels




    def forward(self, x):
        # ----- Encoder -----

        ep = self.pre_convs(x)

        e1 = self.encoder_conv1(ep)
        e1 = F.leaky_relu(e1)

        e1_pooled, idx1 = self.pool1(e1)
        e1_pooled = self.pool1_rev(e1_pooled, idx1)

        encoded = e1_pooled
        # encoded = top_r_channels(e1, self.r_values[0])

        # ------- Decoder -------
        d2 = self.decoder_conv2(encoded)
        d2 = F.leaky_relu(d2)

        dp = self.post_convs(d2)

        return encoded, dp


    def sparsity_loss(self, encoded):
        sparsity_loss = encoded.abs().sum() / (encoded.shape[0] * self.decoded_numel)
        return sparsity_loss * self.sparsity_lambda

    def reconstructive_loss(self, x, decoded):
        return nn.functional.mse_loss(decoded, x, reduction="mean")

    def contrastive_loss(self, encoded):
        encoded_opt, encoded_sub = encoded.chunk(2, dim=0)
        c_f_opt, d_f_opt = encoded_opt.chunk(2, dim=1)
        c_f_sub, d_f_sub = encoded_sub.chunk(2, dim=1)

        c_diff_loss = torch.norm(c_f_opt - c_f_sub, p=1, dim=(1,2,3)).mean()
        d_prod_loss = torch.norm(d_f_opt * d_f_sub, p=1, dim=(1,2,3)).mean()

        contrastive_loss = c_diff_loss + d_prod_loss

        return  contrastive_loss * self.contrastive_lambda