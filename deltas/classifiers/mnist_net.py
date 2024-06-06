import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Simple(nn.Module):
    def __init__(self, final=10):
        self.final = final
        super().__init__()
        self.layer1 = nn.Linear(784, 100)
        self.act1 = nn.ReLU()
        self.output = nn.Linear(100, final)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.act1(self.layer1(x))
        x = self.sigmoid(self.output(x))
        return x
    
    def predict_probs(self, x):
        probs = self.forward(x)
        if self.final == 1:
            probs = torch.concatenate([probs, 1-probs], dim=1)
        return probs
    

class CNN(nn.Module):
    def __init__(self, final):
        super().__init__()
        self.final = final
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, self.final)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output
    
    def predict(self, x):
        x = self.forward(x)
        pred_y = torch.max(x, 1)[1].data.squeeze()
        return pred_y

    def predict_probs(self, x):
        x = self.forward(x)
        # now do softmax
        return x

    
class MnistNet(nn.Module):
    def __init__(self, final=10):
        self.final = final
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, final)
        self.sigmoid = nn.Sigmoid()

    
    def forward(self, x):
        logits = self.logits(x)
        if self.final == 1:
            return F.sigmoid(logits)
        else:
            return F.softmax(logits, dim=1)

    def logits(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x
    
    def predict_probs(self, x):
        probs  = self.forward(x)
        if self.final == 1:
            probs = torch.concatenate([1-probs, probs], dim=1)
        return probs
    
    # def get_projection(self, x):
    #     logits = self.logits(x)
    #     return logits - self._get_bias()

    def get_projection(self, x):
        probs = self.forward(x)
        if probs.shape[1] > 1:
            probs = probs[:, 1]
            probs = probs.unsqueeze(dim=1)
        return probs - self.get_bias()

        # proj = probs - self.get_bias_probs()
        # # print(proj.shape)

        logits = self.logits(x)
        proj = logits - self._get_bias()
        # print(probs)
        # print(logits)
        return proj 
    
    def _get_bias(self):
        return self.fc2.bias
    
    def get_bias(self):
        # return self._get_bias().cpu().detach().numpy()[0]
        return 0.5

    # def predict_probs(self, x):
    #     logits, _ = self.forward(x)
    #     return F.softmax(logits, dim=1)
    
