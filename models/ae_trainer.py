import os
import torch
import torch.nn as nn

from .earlystopping import EarlyStopping
from .ae import AE

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AETrainer:
    def __init__(self,
                 latent_dim: int = 118,
                 lr: float = 1e-1,
                 lr_scheduler: bool = False,
                 es_patience: int = 0,
                 seed: int = 0) -> None:
        self.criterion = nn.MSELoss()
        self.es = EarlyStopping(patience=es_patience,
                                path=os.path.join(CHECKPOINT_PATH, 'ae.pt'), 
                                verbose=True)
        self.ae = AE(latent_dim)
        self.optimizer = torch.optim.Adam(self.ae.parameters(), lr=lr,
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
        x_hat = self.ae(x)
        _mse = self.criterion(x_hat, x)
        return _mse
    
    def step(self, x):
        self.optimizer.zero_grad()
        _loss = self.forward(x)
        _loss.backward()
        self.optimizer.step()
        return _loss
    
    def evaluate(self, test_dl):
        self.ae.eval()
        self.ae.to(device)
        mse = 0.0
        for x in test_dl:
            x = x.to(device)
            _mse = self.forward(x)
            mse += _mse.item()
        return mse / len(test_dl)
    
    def train(self, train_dl, val_dl, epochs=20):
        mse, val_mse = [], []
        self.ae.to(device)
        for epoch in range(epochs):
            self.ae.train()
            run_mse = 0.0
            for x in train_dl:
                x = x.to(device) # GPU
                _mse = self.step(x)
                run_mse += _mse.item() 
            _val_mse = self.evaluate(val_dl)
            mse.append(run_mse / len(train_dl))
            val_mse.append(_val_mse)
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'[MSE: {mse[-1]:.4f}], '
                  f'[Val_MSE: {_val_mse:.4f}], ', flush=True)
            if self.scheduler:
                self.scheduler.step(_val_mse)
            self.es(_val_mse, self.ae)
            if self.es.early_stop:
                print("Early stopping!")
                break
        self.ae.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, 'ae.pt')))
        return self.ae, mse, val_mse
