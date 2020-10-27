"""

Script with helper functions to load data.

Running the file from bash (as follows) plots the various
latent datasets available::

    $ python data.py

"""

import torch
import numpy as np
from torch import tensor as tt
import sklearn.datasets as skd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def float_tensor(X): return torch.tensor(X).float()


def load_real_data(dataset_name):
    '''Loads read world datasets.

    Parameters
    ----------
    dataset_name : str
        One of 'iris' (4D), 'oilflow' (12D), 'gene' (48D), 'mnist' (784D).

    Returns
    -------
    n : int
        Number of datapoints.
    d : int
        Number of dataset dimensions.
    q : None
        Number of latent dimensions (undefined).
    X : None
        Latent data (undefined).
    Y : torch.tensor
        Dataset. Return shape is (n x d).
    labels : numpy.array
        Data categories/classes.

    '''

    if dataset_name == 'iris':
        iris_data = skd.load_iris()
        Y = float_tensor(iris_data.data)
        labels = iris_data.target

    elif dataset_name == 'oilflow':
        Y = float_tensor(np.loadtxt('../data/oil_data.txt'))
        labels = np.loadtxt('../data/oil_labels.txt')
        labels = (labels @ np.diag([1, 2, 3])).sum(axis=1)

    elif dataset_name == 'gene':
        import pandas as pd

        URL = 'https://raw.githubusercontent.com/sods/ods/master/' +\
            'datasets/guo_qpcr.csv'
        gene_data = pd.read_csv(URL, index_col=0)
        Y = float_tensor(gene_data.values)
        raw_labels = np.array(gene_data.index)

        d = dict()
        i = 0
        for label in raw_labels:
            if label not in d:
                d[label] = i
                i += 1
        labels = [d[x] for x in raw_labels]

    elif dataset_name == 'mnist':
        from tensorflow.keras.datasets.mnist import load_data

        (y_train, train_labels), (y_test, test_labels) = load_data()
        labels = np.hstack([train_labels, test_labels])
        n = len(labels)
        Y = np.vstack([y_train, y_test])
        Y = float_tensor(Y.reshape(n, -1))

    else:
        raise NotImplementedError(str(dataset_name) + ' data not implemented')

    n = len(Y)
    d = len(Y.T)
    q = X = None
    return n, d, q, X, Y, labels


def _load_2d_synthetic_latent(latent_data_shape, n_samples=1000):
    '''Creates synthetic latent variables.

    Parameters
    ----------
    latent_data_shape : str
        One of 'blobs', 'noisy_circles', 'make_moons', 'varied', 'normal'.
    n_samples : int
        Number of data points.

    Returns
    -------
    X : numpy.array
        Latent data (shape nx2).
    labels : numpy.array
        Latent data categories/classes.

    '''

    if latent_data_shape == 'blobs':
        return skd.make_blobs(n_samples=n_samples, random_state=42)

    elif latent_data_shape == 'noisy_circles':
        return skd.make_circles(n_samples=n_samples, factor=.5,
                                noise=.05, random_state=42)

    elif latent_data_shape == 'make_moons':
        return skd.make_moons(n_samples=n_samples, noise=.05, random_state=42)

    elif latent_data_shape == 'varied':
        return skd.make_blobs(n_samples=n_samples, random_state=42,
                              cluster_std=[1.0, 2.5, 0.5])

    elif latent_data_shape == 'normal':
        return np.random.normal(size=(n_samples, 2)), \
               np.random.choice([1, 2], n_samples)

    else:
        raise NotImplementedError(str(latent_data_shape) + ' not recognized.')


def _potential_three(z):
    '''Potential 3 from pymc docs (See ref in `_load_2d_weird_latent`).'''
    z = z.T
    w1 = torch.sin(2.*np.pi*z[0]/4.)
    w2 = 3.*torch.exp(-.5*(((z[0]-1.)/.6))**2)
    p = torch.exp(-.5*((z[1]-w1)/.35)**2)
    p = p + torch.exp(-.5*((z[1]-w1+w2)/.35)**2) + 1e-30
    p = -torch.log(p) + 0.1*torch.abs_(z[0])
    return p


def _load_2d_weird_latent(n=300):
    '''Samples the latent variable from pymc's potential three.

    Parameters
    ----------
    n_samples : int
        Number of data points.

    Returns
    -------
    X : numpy.array
        Latent data (shape nx2).

    Notes
    -----
    From [1]_.

    .. [1] https://docs.pymc.io/notebooks/normalizing_flows_overview.html
    '''

    np.random.seed(42)
    Z = np.linspace(-5, 5, 500)
    Z = np.vstack([np.repeat(Z, 500), np.tile(Z, 500)]).T
    p = torch.exp(-_potential_three(tt(Z))).numpy()
    p /= p.sum()

    choice_idx = range(len(p))
    sample_idx = np.random.choice(choice_idx, n, True, p)
    X = tt(Z[sample_idx, :].copy()).float()
    return X


def generate_synthetic_data(n=300, x_type=None, y_type='hi_dim'):
    '''Creates synthetic data set.

    Parameters
    ----------
    n : int
        Number of data points.
    x_type : None or str
        One of 'blobs', 'noisy_circles', 'make_moons', 'varied', 'normal'.
        If None, potential three is used.
    y_type : str
        One of 'lo_dim' (2 planes), 'hi_dim' (6 non-linear functions), or
        'by_cat' where each label corresponds to a different set of functions.
        'by_cat' will only accept the latent type 'normal'.

    Returns
    -------
    n : int
        Number of datapoints.
    d : int
        Number of dataset dimensions.
    q : None
        Number of latent dimensions (undefined).
    X : None
        Latent data (undefined).
    Y : torch.tensor
        Dataset. Return shape is (n x d).
    labels : numpy.array
        Data categories/classes.

    '''

    def err(): return np.random.normal(size=n)*0.05

    if x_type is None:
        X = _load_2d_weird_latent(n)
        labels = None
    else:
        X, labels = _load_2d_synthetic_latent(x_type, n)

    if y_type == 'hi_dim':
        # sample from gp
        Y = float_tensor(np.vstack([
            0.1 * (X[:, 0] + X[:, 1])**2 - 3.5 + err(),
            0.01 * (X[:, 0] + X[:, 1])**3 + err(),
            2 * np.sin(0.5*(X[:, 0] + X[:, 1])) + err(),
            2 * np.cos(0.5*(X[:, 0] + X[:, 1])) + err(),
            4 - 0.1*(X[:, 0] + X[:, 1])**2 + err(),
            1 - 0.01*(X[:, 0] + X[:, 1])**3 + err(),
        ]).T)

    elif y_type == 'by_cat':
        assert x_type == 'normal'

        Y_1 = float_tensor(np.vstack([
            0.1 * (X[:, 0] + X[:, 1])**2 - 3.5 + err(),
            0.01 * (X[:, 0] + X[:, 1])**3 + err(),
            2 * np.sin(0.5*(X[:, 0] + X[:, 1])) + err()
        ]).T)

        Y_2 = float_tensor(np.vstack([
            2 * np.cos(0.5*(X[:, 0] + X[:, 1])) + err(),
            4 - 0.1*(X[:, 0] + X[:, 1])**2 + err(),
            1 - 0.01*(X[:, 0] + X[:, 1])**3 + err()
        ]).T)

        Y = Y_1.clone()
        Y[labels == 2, :] = Y_2[labels == 2, :]

    elif y_type == 'lo_dim':
        Y = 0.1*float_tensor(np.vstack([
                2*X.sum(axis=1)-1 + X.std()*10*err(),
                5*X.sum(axis=1)-3 + X.std()*10*err()]).T)

    d = len(Y.T)
    q = 2
    return n, d, q, X, Y, labels


def check_model(X, Y, Y_recon=None, title=''):
    '''Plots 2d X, Y and reconstructed Ys

    Parameters
    ----------
    X : array_like
    Y : array_like
    Y_recon : array_like or None
        Y_recon is overlaid on Y
    title : str
    '''

    Y = Y[:, :6]
    if Y_recon is not None:
        Y_recon = Y_recon[:, :6]

    fig = plt.figure(figsize=(12, 4))
    for i in range(len(Y.T)):
        ax = fig.add_subplot(1, len(Y.T), i+1, projection='3d')
        ax.scatter(X[:, 0], X[:, 1], Y[:, i])
        if Y_recon is not None:
            ax.scatter(X[:, 0], X[:, 1], Y_recon[:, i])
    plt.suptitle(title)
