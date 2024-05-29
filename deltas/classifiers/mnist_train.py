import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from deltas.classifiers.mnist_net import MnistNet, Simple, CNN
from tqdm import tqdm


class MNIST_torch():
    def __init__(self, hots=1, cuda=False, lr=0.01):
        self.hots = hots
        self.simple = False
        if self.simple == False:
            net = MnistNet(final=self.hots)
            # net = CNN(final=self.hots)
        else:
            net = Simple(final=self.hots)
        if cuda == True:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
        self.net = net.to(self.device)
        # self.optim = optim.Adadelta(self.net.parameters(), lr=lr)
        self.optim = optim.Adam(self.net.parameters(), lr=lr)
        self.scheduler = StepLR(self.optim, step_size=1, gamma=0.7)


    def fit(self, X, y, epochs=5):
        self.train_lm_numpy(X, y, epochs=epochs)
        self.net.eval()
        return self

    
    def train_lm_numpy(self, X, y, epochs=5):
        self.net.train()
        bs = 256
        if self.hots > 1:
            self.loss = nn.CrossEntropyLoss()
        else:
            # self.loss = nn.BCEWithLogitsLoss()
            self.loss = nn.BCELoss()
        num_splits = (len(y)//bs) + 1
        all_ids = np.arange(len(y))
        # for epoch in tqdm(range(epochs), desc='Training model', leave=False):
        for epoch in range(epochs):
            np.random.shuffle(all_ids)
            batch_ids = np.array_split(all_ids, num_splits)
            for i, batch in enumerate(batch_ids):
                if self.simple == True:
                    bX = X[batch]
                else:
                    bX = X[batch].reshape([len(batch), 1, 28, 28])
                data = torch.from_numpy(bX)
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
                loss = self.loss(probs, one_hot)
                # loss = F.nll_loss(probs, t)
                # logits = self.net.logits(data)
                # loss = self.loss(logits, probs)
                loss.backward()
                self.optim.step()
                if i % 100 == 0:
                    # print(f'Train Epoch: {epoch}: {loss.item()}')

                    self.test(X, y, data_s='train')
            # self.scheduler.step()
            # print(f'Train Epoch: {epoch}: {loss.item()}')

    def predict(self, X, bs=2048):
        if self.hots == 1:
            return self.predict_bin(X, bs)
        else:
            return self.predict_multi(X, bs)

        

    def predict_bin(self, X, bs=2048):
        y = self._get_net_func(X, self.net.forward, dims=1, bs=bs)
        return np.round(y)


    def predict_multi(self, X, bs=2048):
        y = self.predict_proba(X, bs=bs)
        return np.argmax(y, axis=1)
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
        # wrapper to work numpy data with torch
        X = X.astype(np.float32)
        self.net.eval()
        y = np.zeros([X.shape[0], dims])
        num_splits = (len(y)//bs) + 1
        all_ids = np.arange(len(y))
        batch_ids = np.array_split(all_ids, num_splits)
        with torch.no_grad():
            for batch in batch_ids:
                # to torch data
                if self.simple == True:
                    bX = X[batch]
                else:
                    bX = X[batch].reshape([len(batch), 1, 28, 28])
                data = torch.from_numpy(bX)
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
                for i, pred in enumerate(preds):
                    if pred == y[batch][i]:
                        correct += 1
        
        print(f'{data_s} set: Accuracy: {correct / len(y)}')

    
    def test_bal(self, X, y, bs=2024, data_s='test'):
        self.net.eval()
        cls, cnts = np.unique(y, return_counts=True)
        correct = {}
        for c in cls:
            correct[c] = 0
        with torch.no_grad():
            num_splits = (len(y)//bs) + 1
            all_ids = np.arange(len(y))
            batch_ids = np.array_split(all_ids, num_splits)
            for batch in batch_ids:
                preds = self.predict(X[batch])
                for i, pred in enumerate(preds):
                    if pred == y[batch][i]:
                        correct[y[batch][i]] += 1
        avg = 0
        for c, cnt in zip(cls, cnts):
            avg += correct[c] / cnt
        
        avg /= len(cls)
        print(f'{data_s} set: Balanced Accuracy: {avg}')
    