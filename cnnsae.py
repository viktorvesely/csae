import torch
import torch.nn as nn
import torch.nn.functional as F

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

class CSAE(nn.Module):
    def __init__(
        self,
        in_channels=256,
        hidden_channels_factors=(10, 10),
        kernel_size=3,
        pooling_size=2,
        groups=1,
        r_values=(10,),  # top-r competition per layer
        sparsity_lambda: float = 1.0,
        contrastive_lambda: float = 1.0
    ):

        print("Factors" , hidden_channels_factors)
        print("topr", r_values)

        super(CSAE, self).__init__()

        self.sparsity_lambda = sparsity_lambda
        self.contrastive_lambda = contrastive_lambda
        hidden_channels = (
            in_channels * hidden_channels_factors[0],
            in_channels * hidden_channels_factors[0] * hidden_channels_factors[1]
        )

        self.encoder_pre_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding= kernel_size // 2
        )

        # ----- Encoder Convolutions -----
        self.encoder_conv1 = nn.Conv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )
        self.pool1 = nn.MaxPool2d(pooling_size, stride=2, return_indices=True)


        self.encoder_conv2 = nn.Conv2d(
            hidden_channels[0],
            hidden_channels[1],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )
        self.pool2 = nn.MaxPool2d(pooling_size, stride=2, return_indices=True)

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.pooling_size = pooling_size
        self.stride = 2
        self.r_values = r_values
        self.groups = groups

        self.unpool2 = nn.MaxUnpool2d(pooling_size, stride=2)
        self.decoder_conv2 = nn.Conv2d(
            hidden_channels[1],
            hidden_channels[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )
        self.unpool1 = nn.MaxUnpool2d(pooling_size, stride=2)

        self.decoder_conv1 = nn.Conv2d(
            hidden_channels[0],
            in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=groups
        )

        self.decoder_post_conv = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding= kernel_size // 2
        )


    def forward(self, x):
        # ----- Encoder -----
        ep = self.encoder_pre_conv(x)
        ep = F.leaky_relu(ep)

        e1 = self.encoder_conv1(ep)
        e1 = F.leaky_relu(e1)

        e1_pooled, idx1 = self.pool1(e1)

        e2 = self.encoder_conv2(e1_pooled)
        e2 = F.leaky_relu(e2)

        e2_pooled, idx2 = self.pool2(e2)

        encoded = top_r_channels(e2_pooled, self.r_values[0])

        # ------- Decoder -------
        d2 = self.unpool2(encoded, idx2, output_size=e2.size())
        d2 = self.decoder_conv2(d2)
        d2 = F.leaky_relu(d2)

        d1 = self.unpool1(d2, idx1, output_size=e1.size())

        d1 = self.decoder_conv1(d1)
        d1 = F.leaky_relu(d1)

        dp = self.decoder_post_conv(d1)

        return encoded, dp


    def sparsity_loss(self, encoded):
        sparsity_loss = encoded.abs().mean()
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