import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from channels.channel_base import Channel  
from models.model_base import BaseModel


class DJSCCN_CIFAR(BaseModel):
    def __init__(self, args, in_channel, class_num):
        super(DJSCCN_CIFAR, self).__init__(args, in_channel, class_num)

        self.channel_type = "AWGN"
        self.base_snr = None
        self.channel = Channel(channel_type="AWGN", snr=args.base_snr)

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1),
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
            nn.Conv2d(in_channels=32, out_channels=args.var_cdim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(args.var_cdim, 64, kernel_size=3, stride=1, padding=1),
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
        print(f"Input size: {x.size()}")  # In kích thước đầu vào

        z = self.encoder(x)
        print(f"Size after encoder: {z.size()}")  # In kích thước sau encoder

        z = self.normalize_layer(z)
        print(f"Size after normalization: {z.size()}")  # In kích thước sau normalize

        z = self.channel(z)
        print(f"Size after channel: {z.size()}")  # In kích thước sau channel

        x_hat = self.decoder(z)
        print(f"Size after decoder: {x_hat.size()}")  # In kích thước sau decoder

        return x_hat

    def get_latent(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc

    def get_train_recon(self, x, base_snr):
        z = self.encoder(x)
        z = self.normalize_layer(z)
        # print("Debug")
        if self.channel is not None:
            z = self.channel(z)
            # print(f"channel: {self.channel.channel_type} {self.channel.snr} | loss: {loss(z, z_z)}")
            # print(f"channel_type: {self.channel.channel_type}, snr: {self.channel.snr}")
        x_hat = self.decoder(z)
        return x_hat

    def get_latent_size(self, x):
        enc = self.encoder(x)
        enc = self.normalize_layer(enc)
        return enc.size()

    def normalize_layer(self, z):
        k = torch.tensor(1.0).to(self.device)  # torch.prod(torch.tensor(z.size()[1:], dtype=torch.float32))
        # Square root of k and P
        sqrt1 = torch.sqrt(k * self.P)

        # Conjugate Transpose of z
        # if torch.is_complex(z):
        #     zT = torch.conj(z).permute(0, 1, 3, 2)
        # else:
        #     zT = z.permute(0, 1, 3, 2)
        # Multiply z and zT = sqrt2
        sqrt2 = torch.sqrt(z*z + self.e)
        div = z / sqrt2
        z_out = div * sqrt1

        return z_out

    def change_channel(self, channel_type='AWGN', snr=None):
        if snr is None:
            self.channel = None
        else:
            self.channel = Channel(channel_type, snr)

    def get_channel(self):
        if hasattr(self, 'channel') and self.channel is not None:
            return self.channel.get_channel()
        return None