import torch
import torch.nn as nn

class CSAE(nn.Module):
    def __init__(
        self,
        in_channels=256,
        ls_factor=5,
        kernel_size=3,
        pooling_size=2,
        r_values=(10,),
        sparsity_lambda: float = 1.0,
        contrastive_lambda: float = 1.0,
        GFC_divisor: int = 1,
        spatial: int = 8
    ):

        print("lsfactor" , ls_factor)

        super(CSAE, self).__init__()

        self.sparsity_lambda = sparsity_lambda
        self.contrastive_lambda = contrastive_lambda
        self.d_channels = in_channels * ls_factor
        self.split_dim = in_channels + (self.d_channels - in_channels) // 2
        p = kernel_size // 2

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, self.split_dim, kernel_size=kernel_size, padding=p, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.split_dim, self.d_channels, kernel_size=kernel_size, padding=p, bias=True),
            nn.LeakyReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(self.d_channels, self.split_dim, kernel_size=kernel_size, padding=p, bias=True),
            nn.LeakyReLU(),
            nn.Conv2d(self.split_dim, in_channels, kernel_size=kernel_size, padding=p, bias=True),
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
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

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