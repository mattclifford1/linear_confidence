import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from deltas.classifiers.mnist_net import MnistNet, Simple, CNN
from deltas.classifiers.base_torch import base_trainer
from deltas.classifiers.large_margin_loss import LargeMarginLoss
from deltas.classifiers.large_margin_net import MnistNetLargeMargin


class MNIST_torch(base_trainer):
    def __init__(self, hots=1, cuda=False, lr=0.01):
        simple = False
        if simple == False:
            net = MnistNet(final=hots)
            # net = CNN(final=self.hots)
        else:
            net = Simple(final=hots)
        if hots > 1:
            self.loss = nn.CrossEntropyLoss()
        else:
            # self.loss = nn.BCEWithLogitsLoss()
            self.loss = nn.BCELoss()
        super().__init__(net=net, hots=hots, cuda=cuda, lr=lr)

    def get_loss_torch(self, X, y):
        # get loss from torch tensor X and labels y
        probs = self.net(X)
        loss = self.loss(probs, y)
        return loss

    
class LargeMarginClassifier(base_trainer):
    def __init__(self, hots=1, cuda=False, lr=0.01):
        net = MnistNetLargeMargin(final=hots)
        self.loss = LargeMarginLoss(
            gamma=10000,
            alpha_factor=4,
            top_k=1,
            dist_norm=np.inf
        )
        super().__init__(net=net, hots=hots, cuda=cuda, lr=lr)

    def get_loss_torch(self, X, y):
        # get loss from torch tensor X and labels y
        probs, feature_maps = self.net.get_probs_and_features(X)
        loss = self.loss(probs, y, feature_maps)
        return loss


    