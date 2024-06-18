import numpy as np

from deltas.classifiers.mimic_net import Simple
from deltas.classifiers.base_torch import base_trainer
from deltas.classifiers.large_margin_loss import LargeMarginLoss

    
class LargeMarginClassifier(base_trainer):
    def __init__(self, hots=1, cuda=False, lr=0.01):
        net = Simple(final=hots)
        self.loss = LargeMarginLoss(
            gamma=10000,
            alpha_factor=4,
            top_k=1,
            dist_norm=np.inf
        )
        super().__init__(net=net, hots=hots, cuda=cuda, lr=lr, images=False)

    def get_loss_torch(self, X, y):
        # get loss from torch tensor X and labels y
        probs, feature_maps = self.net.get_probs_and_features(X)
        loss = self.loss(probs, y, feature_maps)
        return loss


    