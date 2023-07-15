
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Encoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(358, 2 * latent_dims)
        self.linear2 = nn.Linear(2 * latent_dims, latent_dims)
        self.norm = nn.BatchNorm1d(2 * latent_dims)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(self.norm(x))
        return self.linear2(x)
    
class Decoder(nn.Module):
    def __init__(self, latent_dims) -> None:
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 2 * latent_dims)
        self.linear2 = nn.Linear(2 * latent_dims, 358)
        self.norm = nn.BatchNorm1d(2 * latent_dims)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(self.norm(x))
        x = self.linear2(x)
        x = F.tanh(x)
        return x
    
class AE(nn.Module):
    def __init__(self, latent_dims) -> None:
        super(AE, self).__init__()
        self.encoder  = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)