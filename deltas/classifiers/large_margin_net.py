import torch
import torch.nn as nn
import torch.nn.functional as F

    
class MnistNetLargeMargin(nn.Module):
    def __init__(self, final=10):
        super().__init__()
        self.final = final
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(True)
        )
        self.fc2 = nn.Linear(512, final)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)
        fc1 = self.fc1(flatten)
        logits = self.fc2(fc1)
        probs = F.softmax(logits, dim=1)
        return probs
    
    def get_probs_and_features(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)
        fc1 = self.fc1(flatten)
        logits = self.fc2(fc1)
        probs = F.softmax(logits, dim=1)
        return probs, [conv1, conv2]
    
    def logits(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        flatten = conv2.view(x.shape[0], -1)
        fc1 = self.fc1(flatten)
        return self.fc2(fc1)
    
    def predict_probs(self, x):
        probs = self.forward(x)
        if self.final == 1:
            probs = torch.concatenate([probs, 1-probs], dim=1)
        return probs
    
    # def get_projection(self, x):
    #     logits = self.logits(x)
    #     return logits - self._get_bias()

    def get_projection(self, x):
        probs = self.predict_probs(x)
        proj = probs[:, 1]
        return proj.unsqueeze(dim=1)
        # proj = probs - self.get_bias_probs()
        # # print(proj.shape)

        # logits = self.logits(x)
        # probs = F.softmax(logits, dim=1)
        # # print(probs)
        # # print(logits)
        # return proj 
    
    def _get_bias(self):
        return self.fc2.bias
    
    def get_bias(self):
        # return self._get_bias().cpu().detach().numpy()[0]
        return 0.5

    # def predict_probs(self, x):
    #     logits = self.forward(x)
    #     return F.softmax(logits, dim=1)
    
