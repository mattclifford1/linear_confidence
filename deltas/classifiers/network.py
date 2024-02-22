import torch 
import numpy as np


class no_layer_net(torch.nn.Module):
    # Constructor
    def __init__(self, input_size=2, output_size=1):
        super(no_layer_net, self).__init__()
        self.linear_one = torch.nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.linear_one(x))

    def predict_proba(self, X):
        # sklearn style
        X = torch.from_numpy(X).float()
        yhat = self.forward(X)
        yhat = yhat.cpu().detach().numpy()
        return np.hstack([1-yhat, yhat])


# Define the class for single layer NN
class one_layer_net(torch.nn.Module):
    # Constructor
    def __init__(self, input_size=2, hidden_neurons=3, output_size=1):
        super(one_layer_net, self).__init__()
        # hidden layer
        self.linear_one = torch.nn.Linear(input_size, hidden_neurons)
        self.linear_two = torch.nn.Linear(hidden_neurons, output_size)
        # defining layers as attributes
        self.layer_in = None
        self.act = None
        self.layer_out = None
    # prediction function

    def forward(self, x):
        self.layer_in = self.linear_one(x)
        self.act = torch.sigmoid(self.layer_in)
        self.layer_out = self.linear_two(self.act)
        y_pred = torch.sigmoid(self.linear_two(self.act))
        return y_pred
    
    def predict_proba(self, X):
        # sklearn style
        X = torch.from_numpy(X).float()
        print(X.shape)
        yhat = self.forward(X)
        return yhat.cpu().detach().numpy()

