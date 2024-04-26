import torch
import torch.nn as nn
import torch.nn.functional as F

    
class MnistNet(nn.Module):
    def __init__(self, final=10):
        self.final = final
        super(MnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, final)

    def forward(self, x):
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
        # output = F.log_softmax(x, dim=1)
        output = F.softmax(x, dim=1)
        return output
    
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
            probs = torch.concatenate([probs, 1-probs], dim=1)
        return probs
    
    # def get_projection(self, x):
    #     logits = self.logits(x)
    #     return logits - self._get_bias()

    def get_projection(self, x):
        # probs = self.forward(x)
        # proj = probs[:, 1]
        # return proj.unsqueeze(dim=1) - self.get_bias()

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
    
