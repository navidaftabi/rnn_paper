# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 12:01:51 2023

@author: naftabi
"""


import torch
import torch.nn as nn
import torch.distributions
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(358, 2 * latent_dims)
        self.linear2 = nn.Linear(2 * latent_dims, latent_dims)
        self.norm = nn.BatchNorm1d(2 * latent_dims)
        
        self.hidden2mu = nn.Linear(latent_dims, latent_dims)
        self.hidden2log_var = nn.Linear(latent_dims, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        if device == 'cuda':
            self.N.loc = self.N.loc.cuda() # to get sampling on the GPU
            self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(self.norm(x))
        x = self.linear2(x)
        x = F.relu(x)
        
        mu =  self.hidden2mu(x)
        sigma = torch.exp(0.5 * self.hidden2log_var(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 0.5).mean()
        return z
    
class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 2 * latent_dims)
        self.linear2 = nn.Linear(2 * latent_dims, 358)
        self.norm = nn.BatchNorm1d(2 * latent_dims)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(self.norm(x))
        x = self.linear2(x)
        # x = F.tanh(x)
        return x

class VAE(nn.Module):
    def __init__(self, latent_dims):
        super(VAE, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        return self.decoder(self.encoder(x))