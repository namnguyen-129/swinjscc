from tqdm import tqdm
import numpy as np
import os
import glob
import time

import torch
from torch import nn
from torch.optim import Adam

from train.train_base import BaseTrainer
from models.swinjscc import *
from modules.distortion import Distortion

import torch
from torch.optim import Adam
from tqdm import tqdm

from models.swinjscc import SWINJSCC
#from losses import Distortion  # hoặc nơi bạn định nghĩa loss của SwinJSCC

class SWINJSCCTrainer(BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        # Khởi tạo model SwinJSCC với args
        self.model = SWINJSCC(args, self.in_channel, self.class_num).to(self.device)
        self.optimizer = Adam(self.model.parameters(), lr=args.lr)
        self.criterion = Distortion(args)
        self.base_snr = args.base_snr

    def train(self):
        for epoch in range(self.args.out_e):
            self.model.train()
            total_loss = 0.0
            """ Vì HR_Image không trar về tuple gồm ảnh và label mà chỉ trả về ảnh thôi"""
            for batch in tqdm(self.train_dl, desc=f"Epoch {epoch}"):
                if isinstance(batch, (tuple, list)) and len(batch) == 2:
                    images, _ = batch
                else:
                    images = batch
                images = images.to(self.device)
                # Forward
                recon, CBR, SNR, mse_per_pixel, loss_G = self.model(images)
                loss = loss_G

                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_dl)
            self.writer.add_scalar('train/loss', avg_loss, epoch)
            print(f"[Train] Epoch {epoch}: loss = {avg_loss:.4f}")

            # Validation
            val_loss = self.evaluate_epoch()
            self.writer.add_scalar('val/loss', val_loss, epoch)
            print(f"[Val]   Epoch {epoch}: loss = {val_loss:.4f}")

            self.save_model(epoch=epoch, model=self.model)

        self.writer.close()
        self.save_config()

    def test(self):
        # Gọi evaluate từ BaseTrainer
        self.evaluate(self.args.config_path, self.args.out)



# class SWINJSCCLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.rec_loss = nn.MSELoss()
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=0)

#     def forward(self, args, rec, img):
#         rec_loss = self.rec_loss(rec, img)
#         total_loss = args.rec_coeff * rec_loss

#         return total_loss
