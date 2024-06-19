import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Simple(nn.Module):
    def __init__(self, input=19, final=1):
        self.final = final
        super().__init__()
        self.layer1 = nn.Linear(input, 100)
        self.act1 = nn.ReLU()
        self.layer2 = nn.Linear(100, 500)
        self.act2 = nn.ReLU()
        self.layer3 = nn.Linear(500, 100)
        self.act3 = nn.ReLU()
        self.output = nn.Linear(100, final)
        self.sigmoid = nn.Sigmoid()

    def logits(self, x):
        x = self.act1(self.layer1(x))
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        return self.output(x)

    def forward(self, x):
        logits = self.logits(x)
        if self.final == 1:
            return F.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)
    
    def predict_probs(self, x):
        probs = self.forward(x)
        if self.final == 1:
            probs = torch.concatenate([1-probs, probs], dim=1)
        return probs
    
    def get_bias(self):
        # return self._get_bias().cpu().detach().numpy()[0]
        return 0.5
    
    def get_projection(self, x):
        probs = self.forward(x)
        if probs.shape[1] > 1:
            probs = probs[:, 1]
            probs = probs.unsqueeze(dim=1)
        return probs - self.get_bias()
    
    # def get_probs_and_features(self, x):
    #     x1 = self.act1(self.layer1(x))
    #     x2 = self.act2(self.layer2(x1))

    #     logits = self.output(x2)

    #     probs = F.softmax(logits, dim=1)
    #     return probs, [x1, x2]
    
