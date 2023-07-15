# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 18:20:08 2023

@author: naftabi
"""

import os
import torch
import torch.nn as nn

from .earlystopping import EarlyStopping
from .rnn import RNN


# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class RNNTrainer:
    def __init__(self, 
                 input_dim: int = 118,
                 outputdims: int = 118,
                 hidden_dim: int = 100,
                 num_layers: int = 1,
                 lr: float = 1e-1,
                 lr_scheduler: bool = False,
                 es_patience: int = 0,
                 seed: int = 0):
        self.set_seed(seed)
        self.criterion = nn.MSELoss()
        self.es = EarlyStopping(patience=es_patience,
                                path=os.path.join(CHECKPOINT_PATH, 'rnn.pt'), 
                                verbose=True)
        self.rnn = RNN(input_dims=input_dim,
                       outputdims=outputdims,
                       hidden_size=hidden_dim,
                       num_layers=num_layers)
        self.optimizer = torch.optim.Adam(self.rnn.parameters(), lr=lr,
                                          weight_decay=5e-5)
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

    def forward(self, x, y):
        y_hat = self.rnn(x)
        return self.criterion(y_hat, y)
    
    def step(self, x, y):
        self.optimizer.zero_grad()
        _loss = self.forward(x, y)
        _loss.backward()
        self.optimizer.step()
        return _loss

    def evaluate(self, test_dl):
        self.rnn.eval()
        self.rnn.to(device)
        mse = 0.0
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            _loss = self.forward(x, y)
            mse += _loss.item()
        return mse / len(test_dl)
    
    def train(self, train_dl, test_dl, epochs=20):
        mse = []
        val_mse = []
        self.rnn.to(device)
        for epoch in range(epochs):
            self.rnn.train()
            run_mse = 0.0
            for x, y in train_dl:
                x = x.to(device) # GPU
                y = y.to(device)
                _loss = self.step(x, y)
                run_mse += _loss.item()
            val_loss = self.evaluate(test_dl)
            mse.append(run_mse / len(train_dl))
            val_mse.append(val_loss)
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'[MSE: {mse[-1]:.4f}], '
                  f'[Val_MSE: {val_loss:.2f}], ', flush=True)
            if self.scheduler:
                self.scheduler.step(val_loss)
            self.es(val_loss, self.rnn)
            if self.es.early_stop:
                print("Early stopping!")
                break
        self.rnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, 'rnn.pt')))
        return self.rnn, mse, val_mse