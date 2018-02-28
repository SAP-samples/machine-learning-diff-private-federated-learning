import os
import struct
import numpy as np
import pickle

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""

def read(dataset = "training", path = "."):

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError, "dataset must be 'testing' or 'training'"

    print(fname_lbl)

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)


    # Reshape and normalize

    img = np.reshape(img, [img.shape[0], img.shape[1]*img.shape[2]])*1.0/255.0

    return img, lbl


def get_data(d):
    # load the data
    x_train, y_train = read('training', d + '/MNIST_original')
    x_test, y_test = read('testing', d + '/MNIST_original')

    # create validation set
    x_vali = list(x_train[50000:].astype(float))
    y_vali = list(y_train[50000:].astype(float))

    # create test_set
    x_train = x_train[:50000].astype(float)
    y_train = y_train[:50000].astype(float)

    # sort test set (to make federated learning non i.i.d.)
    indices_train = np.argsort(y_train)
    sorted_x_train = list(x_train[indices_train])
    sorted_y_train = list(y_train[indices_train])

    # create a test set
    x_test = list(x_test.astype(float))
    y_test = list(y_test.astype(float))

    return sorted_x_train, sorted_y_train, x_vali, y_vali, x_test, y_test


class Data:
    def __init__(self, save_dir, n):
        raw_directory = save_dir + '/DATA'
        self.client_set = pickle.load(open(raw_directory + '/clients/' + str(n) + '_clients.pkl', 'rb'))
        self.sorted_x_train, self.sorted_y_train, self.x_vali, self.y_vali, self.x_test, self.y_test = get_data(save_dir)
