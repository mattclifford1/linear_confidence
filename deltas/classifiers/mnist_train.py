import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from deltas.classifiers.mnist_net import MnistNet


class MNIST_torch():
    def __init__(self, binary=True):
        if binary == True:
            net = MnistNet(final=2)
            self.hots = 2
        else:
            net = MnistNet(final=10)
            self.hots = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = net.to(self.device)
        self.optim = optim.Adadelta(self.net.parameters(), lr=1.0)
        self.scheduler = StepLR(self.optim, step_size=1, gamma=0.7)


    def fit(self, X, y, epochs=5):
        self.train_lm_numpy(X, y, epochs=epochs)
        self.net.eval()
        return self

    
    def train_lm_numpy(self, X, y, epochs=5):
        self.net.train()
        bs = 256
        self.loss = nn.CrossEntropyLoss()
        num_splits = (len(y)//bs) + 1
        all_ids = np.arange(len(y))
        for epoch in range(epochs):
            np.random.shuffle(all_ids)
            batch_ids = np.array_split(all_ids, num_splits)
            for i, batch in enumerate(batch_ids):
                data = torch.from_numpy(
                    X[batch].reshape([len(batch), 1, 28, 28])
                )
                data = data.to(self.device)
                target = torch.from_numpy(y[batch])
                if self.hots != 1:
                    one_hot = torch.zeros(len(target), self.hots).scatter_(
                        1, target.unsqueeze(1), 1.).float()
                    one_hot = one_hot.to(self.device)
                else:
                    one_hot = target.float().to(self.device)
                self.optim.zero_grad()
                # t = target.to(self.device)
                probs = self.net(data)
                # loss = F.nll_loss(probs, t)
                loss = self.loss(probs, probs)
                loss.backward()
                self.optim.step()
                # if i % 100 == 0:
                #     print(f'Train Epoch: {epoch}: {loss.item()}')
            self.scheduler.step()
            # print(f'Train Epoch: {epoch}: {loss.item()}')


    def predict(self, X, bs=2048):
        y = self.predict_proba(X, bs=bs)
        y = np.argmax(y, axis=1)
        return y
        # y = y[:, 1]
        # y[y<0.5] = 0
        # y[y>=0.5] = 1
        # # return y
        # proj = self.get_projection(X, bs=bs).squeeze()
        # proj[proj<0.5] = 0
        # proj[proj>=0.5] = 1
        # return proj.squeeze()


    def predict_proba(self, X, bs=2048):
        if self.hots == 1:
            dims = 2
        else:
            dims = self.hots
        return self._get_net_func(X, self.net.predict_probs, dims=dims, bs=bs)
    

    def get_projection(self, X, bs=2048):
        # minority_class = 1
        # proj = np.expand_dims(self._get_net_func(X, self.net.get_projection, dims=self.hots, bs=bs)[:, minority_class], axis=1)
        # proj = np.expand_dims(self._get_net_func(X, self.net.get_projection, dims=1, bs=bs), axis=1)
        proj = self._get_net_func(X, self.net.get_projection, dims=1, bs=bs)
        return proj
    
    def get_bias(self):
        return self.net.get_bias()
    
    
    def _get_net_func(self, X, func, dims, bs=2048):
        X = X.astype(np.float32)
        self.net.eval()
        y = np.zeros([X.shape[0], dims])
        num_splits = (len(y)//bs) + 1
        all_ids = np.arange(len(y))
        batch_ids = np.array_split(all_ids, num_splits)
        with torch.no_grad():
            for batch in batch_ids:
                # to torch data
                data = torch.from_numpy(
                    X[batch].reshape([len(batch), 1, 28, 28])
                )
                data = data.to(self.device)
                # predict
                output = func(data)
                # save to return
                y[batch, :] = output.cpu().numpy()
        return y


    def test(self, X, y, bs=2024, data_s='test'):
        self.net.eval()
        correct = 0
        with torch.no_grad():
            num_splits = (len(y)//bs) + 1
            all_ids = np.arange(len(y))
            batch_ids = np.array_split(all_ids, num_splits)
            for batch in batch_ids:
                preds = self.predict(X[batch])
                correct += (preds == y[batch]).sum().item()

        print('{} set: Accuracy: {}/{} ({:.0f}%)'.format(
            data_s, correct, len(y),
            100. * correct / len(y)))
    