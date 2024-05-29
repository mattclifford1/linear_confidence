import numpy as np
from tqdm import tqdm
from torchvision import transforms
from torchvision import datasets
from torch.utils import data as torch_data
import deltas
import os


def get_mnist(size=60000, minority_id=[0], binary=True, classes_remove=[], ratio=False, split=0.1, equal_test=False, seed=True):  # 60000 is full dataset
    download_dir = os.path.join(os.path.expanduser('~'), 'Downloads')
    train_loader = torch_data.DataLoader(
        datasets.MNIST(download_dir, train=True, download=True,
                    transform=transforms.Compose([
                        transforms.ToTensor(),
                        # transforms.Normalize((0.1307,), (0.3081,))
                    ])),
        batch_size=256, shuffle=False, drop_last=False)

    # test_loader = torch_data.DataLoader(
    #     datasets.MNIST(download_dir, train=False,
    #                 transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.Normalize((0.1307,), (0.3081,))
    #                 ])),
    #     batch_size=2048, shuffle=False, drop_last=False)


    
    X = np.empty([size, 28*28], dtype=np.float32)
    # X = np.empty([size, 1, 28, 28], dtype=np.float32)
    y = np.zeros(size, dtype=np.int64)
    counter = 0
    for batch_idx, (d, t) in tqdm(enumerate(train_loader), total=size//256, desc='torch to numpy MNIST', leave=False):
        d_np = d.numpy()
        # t_np = t.numpy()
        for i in range(d.shape[0]):
            X[counter, :] = d_np[i, :, :, :].reshape(-1)
            # X[counter, :, :, :] = d_np[i, :, :, :]
            y[counter] = t[i]
            counter += 1
            if counter == size:
                break
        if counter == size:
            break
    
    # remove some classes
    for cls in classes_remove:
        X = np.delete(X, np.where(y==cls)[0], axis=0)
        y = np.delete(y, np.where(y==cls)[0])
    # convert data to binary
    if binary == True:
        for id in minority_id:
            y[y == id] = 11  # take out of range of labels
        y[y != 11] = 0
        y[y == 11] = 1
    # formatting
    print(len(y), sum(y))
    data = {'X':X, 'y':y}
    # shuffle the dataset
    data = deltas.data.utils.shuffle_data(data, seed=seed)
    # split into train, test
    train_data, test_data = deltas.data.utils.proportional_split(
        data, size=split, ratio=ratio, equal_test=equal_test, seed=seed)
    return train_data, test_data


if __name__ == '__main__':
    data, test = get_mnist()
    print(data['X'].shape)
    print(test['X'].shape)
