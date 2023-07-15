import os
import torch
import torch.nn as nn

from .earlystopping import EarlyStopping
from .dnn import DNN


# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.path.join(os.getcwd(), 'checkpoints')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class DNNTrainer:
    def __init__(self,
                 lr: float = 1e-1,
                 lr_scheduler: bool = False,
                 es_patience: int = 0,
                 seed: int = 0):
        self.set_seed(seed)
        self.criterion = nn.CrossEntropyLoss()
        self.es = EarlyStopping(patience=es_patience,
                                path=os.path.join(CHECKPOINT_PATH, 'dnn.pt'), 
                                verbose=True)
        self.dnn = DNN()
        self.optimizer = torch.optim.RMSprop(self.dnn.parameters(), lr=lr, weight_decay=5e-5, momentum=5e-3)
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

    @staticmethod
    def accuracy(y_hat, y):
        return ((torch.argmax(y_hat, dim=1) == torch.argmax(y, dim=1)).float().sum()) / len(y)
    
    def forward(self, x, y):
        y_hat = self.dnn(x)
        _acc = self.accuracy(y_hat, y)
        _loss = self.criterion(y_hat, y)
        return _loss, _acc
    
    def step(self, x, y):
        self.optimizer.zero_grad()
        _loss, _acc = self.forward(x, y)
        _loss.backward()
        self.optimizer.step()
        return _loss, _acc

    def evaluate(self, test_dl):
        self.dnn.eval()
        self.dnn.to(device)
        loss = 0.0
        acc = 0.0
        for x, y in test_dl:
            x = x.to(device)
            y = y.to(device)
            _loss, _acc = self.forward(x, y)
            loss += _loss.item()
            acc += _acc.item()
        return loss / len(test_dl), acc / len(test_dl)
    
    def train(self, train_dl, val_dl, epochs=20):
        loss = []
        acc = []
        val_acc = []
        val_loss = []
        self.dnn.to(device)
        for epoch in range(epochs):
            self.dnn.train()
            run_loss = 0.0
            run_acc = 0.0
            for x, y in train_dl:
                x = x.to(device) # GPU
                y = y.to(device)
                _loss, _acc = self.step(x, y)
                run_loss += _loss.item()
                run_acc += _acc.item()
            _val_loss, _val_acc = self.evaluate(val_dl)
            acc.append(run_acc / len(train_dl))
            loss.append(run_loss / len(train_dl))
            val_loss.append(_val_loss)
            val_acc.append(_val_acc)
            print(f'Epoch {epoch+1}/{epochs}: '
                  f'[Loss: {loss[-1]:.4f}], '
                  f'[Accuracy: {acc[-1]:.4f}], '
                  f'[Val_Loss: {_val_loss:.4f}], '
                  f'[Val_Accuracy: {_val_acc:.4f}]', flush=True)
            if self.scheduler:
                self.scheduler.step(_val_loss)
            self.es(_val_loss, self.dnn)
            if self.es.early_stop:
                print("Early stopping!")
                break
        self.dnn.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, 'dnn.pt')))
        return self.dnn, loss, val_loss, acc, val_acc