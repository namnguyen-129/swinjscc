import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from models.model_base import BaseModel


class DGSC_CIFAR(BaseModel):
    def __init__(self, args, in_channel, class_num):
        super(DGSC_CIFAR, self).__init__(args, in_channel, class_num)

        self.encoder = nn.Sequential(
            nn.Conv2d(self.in_channel, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(in_channels=32, out_channels=self.var_cdim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(self.var_cdim, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, in_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.SELU(),
        )

    def forward(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        rec = self.decoder(enc)
        return rec

    def get_latent(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc

    def get_semcom_recon(self, x, n_var, channel, device):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        """ Generate Propagating Noise"""
        if channel == "Gaussian":
            pass
        elif channel == "Rayleigh":
            pass
        elif channel == "Rician":
            pass
        noise = torch.normal(mean=torch.zeros(enc.size()),
                             std=torch.ones(enc.size()) * n_var).to(device)
        """ Simulate a noise by physical channels """
        var = enc + noise

        rec = self.decoder(var)
        # print(f"ori: {F.mse_loss(x, rec)}")
        return rec

    def get_latent_size(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc.size()

    def normalize_layer(self, z):
        k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
        # Square root of k and P
        sqrt1 = torch.sqrt(k * self.P)

        # Conjugate Transpose of z
        if torch.is_complex(z):
            zT = torch.conj(z).permute(0, 1, 3, 2)
        else:
            zT = z.permute(0, 1, 3, 2)
        # Multiply z and zT = sqrt2
        sqrt2 = torch.sqrt(zT*z + self.e)
        div = z / sqrt2
        z_out = div * sqrt1

        return z_out  # Adjusted return value as per PyTorch operations

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None