# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:46:46 2023

@author: naftabi
"""

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_dims, outputdims, hidden_size, num_layers=1):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dims, 
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            proj_size=int(hidden_size/2),
                            batch_first=True)
        self.linear = nn.Linear(int(hidden_size/2), outputdims) 
        
    def forward(self, x):
        x, _ = self.lstm(x)
        return self.linear(x)