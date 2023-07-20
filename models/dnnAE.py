import torch
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, no_residuals: int = 1):
        """
        1- no_residuals = 1 if only using autoencoder residuals
        2- no_residuals = 2 if using autoencoder+rnn residuals
        """
        super(DNN, self).__init__()
        # signal features
        self.linear1 = nn.Linear(358, 128)
        self.linear2 = nn.Linear(128, 64)
        self.linear3 = nn.Linear(64, 32)
        # residual features
        self.linear4 = nn.Linear(no_residuals*32, 16) 
        # final layer
        self.linear5 = nn.Linear(16, 4)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()
        self.linear3.reset_parameters()
        self.linear4.reset_parameters()
        self.linear5.reset_parameters()
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        # flatten no_residuals*32
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear4(x))
        # last layer
        x = F.softmax(self.linear6(x), dim=2)
        return x