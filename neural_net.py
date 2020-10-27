import torch
from torch import nn
from torch import optim
from torch.nn import Module, Linear, ReLU, MSELoss, Softplus, Sequential, Sigmoid, Tanh
from torch.utils.data import DataLoader, Dataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class MBSplitter(Dataset):
    set_seed(49)

    def __init__(self, X, y):
        super(MBSplitter, self).__init__()

        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.y)


class Net(nn.Module):

    def __init__(self, inp_dim_main=None, inp_dim_var=None, hidden_dim=None, lmb=0.1, batch_size=320000):

        set_seed(42)

        super().__init__()

        self.net_main = Sequential(Linear(inp_dim_main, hidden_dim),
                                   ReLU(),
                                   Linear(hidden_dim, 1))

        self.net_var = Sequential(Linear(inp_dim_var, hidden_dim),
                                  ReLU(),
                                  Linear(hidden_dim, 1),
                                  Softplus())

        self.lmb = lmb
        self.batch_size = batch_size

    def prepare_input(self, X, y):

        if y.ndim == 1:
            y = y.reshape(-1, 1)

        X = torch.from_numpy(X.astype('float32')).cuda()
        y = torch.from_numpy(y.astype('float32')).cuda()

        return X, y

    def forward(self, X_main, X_var):

        pred = self.net_main(X_main)
        sigma2 = self.net_var(X_var)

        return pred, sigma2

    def loss(self, pred, sigma2, true):

        loss = (sigma2.log() + (true - pred) ** 2 / sigma2).mean()

        return loss

    def get_mimi_bathes(self, X, y):

        X_data = MBSplitter(X, y)
        mb = DataLoader(X_data, batch_size=self.batch_size, shuffle=True)

        return mb

    def fit(self, X_main, X_var, y, n_epochs=100, lr=0.001):

        _ = y.copy()
        X_main, y = self.prepare_input(X_main, y)
        X_var, y = self.prepare_input(X_var, _)

        optimizer = optim.Adam(self.parameters(), weight_decay=self.lmb, lr=lr)

        for epoch in range(n_epochs):

            mb = self.get_mimi_bathes(X_main, y)

            y_out, sigma2 = self.forward(X_main, X_var)
            loss = self.loss(y_out, sigma2, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 500 == 0:
                pass
                # print(loss.item())

        return None

    def predict(self, X_main, X_var):

        X_main = torch.from_numpy(X_main.astype('float32')).cuda()
        X_var = torch.from_numpy(X_var.astype('float32')).cuda()

        pred, sigma2 = self.forward(X_main, X_var)

        return pred.cpu().detach().numpy()

    def get_confidence(self, X_main, X_var):

        X_main = torch.from_numpy(X_main.astype('float32')).cuda()
        X_var = torch.from_numpy(X_var.astype('float32')).cuda()

        pred, sigma2 = self.forward(X_main, X_var)

        return sigma2.cpu().detach().numpy()