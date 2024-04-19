import numpy as np
import torch
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils import data as torch_data
import deltas
from deltas.pipeline import data, classifier, evaluation
from deltas.classifiers.large_margin_train import LargeMarginNet

# data_clf = data.get_real_dataset('Iris', scale=False)

train_loader = torch_data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    # batch_size=256, shuffle=True, drop_last=True)
    batch_size=256, shuffle=False, drop_last=True)

test_loader = torch_data.DataLoader(
    datasets.MNIST('./data', train=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=2048, shuffle=False, drop_last=False)


minority_id = 0
X = np.empty([60000, 28*28], dtype=np.float32)
# X = np.empty([60000, 1, 28, 28], dtype=np.float32)
y = np.zeros(60000, dtype=np.int64)
counter = 0
for batch_idx, (d, t) in tqdm(enumerate(train_loader), total=60000//256, desc='torch to numpy MNIST', leave=False):
    d_np = d.numpy()
    t_np = t.numpy()
    for i in range(d.shape[0]):
        X[counter, :] = d[i, :, :, :].reshape(-1)
        # X[counter, :, :, :] = d_np[i, :, :, :]
        if t[i] == minority_id:
            y[counter] = 1
        # y[counter] = t_np[i]
        counter += 1


a=LargeMarginNet().fit(X, y)
b=LargeMarginNet().train(train_loader, test_loader)


from deltas.classifiers.large_margin_train import test
test(a.net, test_loader)
test(b.net, test_loader)