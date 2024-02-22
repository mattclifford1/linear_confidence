import torch
import deltas.classifiers.network as network
import random
from tqdm import tqdm

def get_model(data):
    model = network.no_layer_net()
    model = loop(model, data)
    return model


def criterion(y_pred, y):
    out = -torch.mean(y*torch.log(y_pred)+ 
                          (1-y)*torch.log(1-y_pred))
    return out

def loop(model, data):
    X = torch.from_numpy(data['X']).float()
    y = torch.from_numpy(data['y']).float()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    epochs = 10
    for epoch in tqdm(range(epochs)):

        # one instance at a time
        inds = list(range(len(y)))
        random.shuffle(inds)
        for i in range(len(y)):
            yhat = model(X[inds[i], :])
            loss = criterion(yhat, y[inds[i]])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        # # all at once
        # yhat = model(X)
        # loss = criterion(yhat, y)
        # loss.backward()
        # optimizer.step()
        # optimizer.zero_grad()
    
    return model
    
