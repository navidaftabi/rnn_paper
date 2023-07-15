# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:19:10 2023

@author: naftabi
"""

import os
import torch
import torch.nn as nn

from .earlystopping import EarlyStopping
from .vae import VAE


# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VAETrainer:
    def __init__(self, 
                 latent_dim: int = 118,
                 lr: float = 1e-1,
                 lr_scheduler: bool = False,
                 es_patience: int = 0,
                 seed: int = 0) -> None:
        self.set_seed(seed)
        self.criterion = nn.MSELoss()
        self.es = EarlyStopping(patience=es_patience,
                                path=os.path.join(CHECKPOINT_PATH, 'vae.pt'), 
                                verbose=True)
        self.vae = VAE(latent_dim)
        self.optimizer = torch.optim.Adam(self.vae.parameters(), lr=lr,
                                          amsgrad=True, weight_decay=5e-5)
        if lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                        mode='min',
                                                                        factor=0.2,
                                                                        patience=6,
                                                                        min_lr=5e-5)
    
    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        print("Device: ", device)

    def forward(self, x):
        x_hat = self.vae(x)
        _kl = self.vae.encoder.kl
        _mse = self.criterion(x_hat, x)
        return _mse, _kl
    
    def step(self, x):
        self.optimizer.zero_grad()
        _mse, _kl = self.forward(x)
        _loss = _mse + _kl
        _loss.backward()
        self.optimizer.step()
        return _mse, _kl
    
    def evaluate(self, test_dl):
        self.vae.eval()
        self.vae.to(device)
        mse, kl = 0.0, 0.0
        for x in test_dl:
            x = x.to(device)
            _mse, _kl = self.forward(x)
            mse += _mse.item()
            kl += _kl.item()
        return mse / len(test_dl), kl / len(test_dl)
    
    def train(self, train_dl, val_dl, epochs=20):
        mse, kl = [], []
        val_mse, val_kl = [], []
        self.vae.to(device)
        for epoch in range(epochs):
            self.vae.train()
            run_mse, run_kl = 0.0, 0.0
            for x in train_dl:
                x = x.to(device) # GPU
                _mse, _kl = self.step(x)
                run_mse += _mse.item() 
                run_kl += _kl.item()
            _val_mse, _val_kl = self.evaluate(val_dl)
            mse.append(run_mse / len(train_dl))
            kl.append(run_kl / len(train_dl))
            val_mse.append(_val_mse)
            val_kl.append(_val_kl)
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'[MSE: {mse[-1]:.4f}], '
                  f'[KL: {kl[-1]:.4f}], '
                  f'[Val_MSE: {_val_mse:.4f}], '
                  f'[Val_KL: {_val_kl:.4f}], ', flush=True)
            if self.scheduler:
                self.scheduler.step(_val_mse)
            self.es(_val_mse, self.vae)
            if self.es.early_stop:
                print("Early stopping!")
                break
        self.vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, 'vae.pt')))
        return self.vae, mse, kl, val_mse, val_kl