# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:54:32 2023

@author: naftabi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # signal features
        self.linear1 = nn.Linear(358, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        # residual features
        self.linear4 = nn.Linear(3*32, 64)
        self.linear5 = nn.Linear(64, 16)
        # final layer
        self.linear6 = nn.Linear(16, 4)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()
        self.linear4.reset_parameters()
        self.linear5.reset_parameters()
        self.linear6.reset_parameters()
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # flatten 3*32
        x = torch.flatten(x, start_dim=2)
        x = F.relu(self.linear4(x))
        x = F.relu(self.linear5(x))
        # last layer
        x = F.softmax(self.linear6(x), dim=2)
        x = torch.mean(x, dim=1)
        return torch.squeeze(x, dim=1)