import numpy as np
from torch.optim import Adam
import torch

from deltas.classifiers.large_margin_loss import LargeMarginLoss
from deltas.classifiers.large_margin_net import MnistNet, MnistNetBin


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# lm = LargeMarginLoss(
#     gamma=10000,
#     alpha_factor=4,
#     top_k=1,
#     dist_norm=np.inf
# )


# def train_lm(model, train_loader, optimizer, epoch, lm):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data = data.to(device)
#         one_hot = torch.zeros(len(target), 10).scatter_(
#             1, target.unsqueeze(1), 1.).float()
#         one_hot = one_hot.cuda()
#         optimizer.zero_grad()
#         output, feature_maps = model(data)
#         # loss = F.mse_loss(output, target) * 5e-4 # l2_loss_weght
#         loss = lm(output, one_hot, feature_maps)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 100 == 0:
#             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))



def train_lm_numpy(model, X, y, optimizer, epoch, LML, device, hots=2):
    model.train()
    bs = 256
    for batch_idx in enumerate(range(len(y)//bs)):
        i = batch_idx[0]
        data = torch.from_numpy(
            X[i*bs:(i+1)*bs].reshape([bs, 1, 28, 28])
            )
        data = data.to(device)

        target = torch.from_numpy(y[i*bs:(i+1)*bs])
        one_hot = torch.zeros(len(target), hots).scatter_(
            1, target.unsqueeze(1), 1.).float()
        one_hot = one_hot.to(device)
        optimizer.zero_grad()
        output, feature_maps = model(data)
        # loss = F.mse_loss(output, target) * 5e-4 # l2_loss_weght
        loss = LML(output, one_hot, feature_maps)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f'Train Epoch: {epoch}: {loss.item()}')


class LargeMarginClassifier():
    def __init__(self, net_type='Mnist-Binary'):
        if net_type == 'Mnist-Binary':
            net = MnistNetBin
        else:
            net = MnistNet
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.LML = LargeMarginLoss(
            gamma=10000,
            alpha_factor=4,
            top_k=1,
            dist_norm=np.inf
        )
        self.net = net().to(self.device)
        self.optim = Adam(self.net.parameters())

    def fit(self, X, y, epochs=5):
        for i in range(0, epochs):
            train_lm_numpy(self.net, X, y, self.optim, i, self.LML, self.device)
        self.net.eval()
        return self
    
    def predict(self, X):
        self.net.eval()
        with torch.no_grad():
            return self._predict(X)

    def _predict(self, X):
        bs = 2048
        y = np.zeros(X.shape[0], dtype=np.int64)
        counter = 0
        for batch_idx in enumerate(range(X.shape[0]//bs)):
            i = batch_idx[0]
            data = torch.from_numpy(
                X[i*bs:(i+1)*bs].reshape([bs, 1, 28, 28])
            )
            data = data.to(self.device)
            output = self.net.predict_probs(data)
            # output = output.numpy()
            for j in range(output.shape[0]):
                y[counter] = torch.argmax(output[j, :])
                counter += 1


        # final left over batch
        left_over = len(y) - ((i+1)*bs)
        data = torch.from_numpy(
            X[(i+1)*bs:].reshape([left_over, 1, 28, 28])
        )
        data = data.to(self.device)
        output = self.net.predict_probs(data)
        # output = output.numpy()
        for j in range(output.shape[0]):
            y[counter] = torch.argmax(output[j, :])
            print(output[j, :])
            counter += 1
        return y
        

    def predict_proba(self, X):
        pass

    def get_projection(self, X):
        pass
    
    # def train(self, train_loader, test_loader):
    #     for i in range(0, 5):
    #         train_lm(self.net, train_loader, self.optim, i, self.LML)

    #         # test(self.net, test_loader)
    #     return self

def test(model, test_loader, device):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, *_ = model(data)
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            _, idx = output.max(dim=1)
            correct += (idx == target).sum().item()

    print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
        correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

