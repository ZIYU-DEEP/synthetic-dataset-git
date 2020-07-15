from math import sin, cos
from abc import ABC, abstractmethod
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import torch
import joblib
import numpy as np


# #########################################################################
# Helper Functions
# #########################################################################
def gen_gaussian_train(normal_mu, abnormal_mu, ratio_abnormal, split, random_state):
    """
    Get the data and scaler used for training the model.

    Inputs:
        normal_mu: (str) in a format like '1_1_1', indicating the mean for the normal
        abnormal_mu: (str) in a format like '1_1_1', indicating the mean for the abnormal
        ratio_abnormal: (float) the ratio for abnormal data / normal data
        split: (float) the ratio for test / train
        random_state: (int) the seed for randomness

    Return:
        X_train, X_test: (np.array) with a shape of (N_instances, 3)
        y_train, y_test: (np.array) with a shape of (N_instances,)
        scaler: (sklearn model) can be used later to scale evaluation data
    """
    # Set random seed
    np.random.seed(random_state)

    # Initialize for Gaussians. Note that we have fixed the covariance.
    cov = [[0.1, 0, 0],
           [0, 0.1, 0],
           [0, 0, 0.1]]
    normal_mu = [int(i) for i in normal_mu.split('_')]

    # Generate X_normal
    X_normal = np.random.multivariate_normal(normal_mu, cov, 6000)
    y_normal = np.zeros(X_normal.shape[0])

    # Generate X_abnormal and concatenate data
    if abnormal_mu:
        # Generate X_abnormal
        abnormal_mu = [float(i) for i in abnormal_mu.split('_')]
        X_abnormal = np.random.multivariate_normal(abnormal_mu, cov, int(6000 * ratio_abnormal))
        y_abnormal = np.ones(X_abnormal.shape[0])

        # Concatenate
        X = np.vstack((X_normal, X_abnormal))
        y = np.hstack((y_normal, y_abnormal))
    else:
        # No need X_abnormal
        X = X_normal
        y = y_normal

    # Do train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=split,
                                                        random_state=random_state,
                                                        stratify=y)

    # Scale the data
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler


def gen_ball(r, mu):
    """
    Generate a set of data points surrounding a point like a ball.
    This function will later be used in main to generate a set of means
    for abnormal test.

    Inputs:
        r: (float) distance between the trained normal and the trained
           abnormal; used as the radius here
        mu: (np.array) a 3d array specifying the mu for trained normal
            or the trained abnormal data

    Returns:
        result: (list) a list a 3d arrays indicating the mean for abnormal
                data to test
    """
    thetas = range(0, 360, 60)
    phis = range(0, 360, 60)
    pairs = [(theta, phi) for theta in thetas for phi in phis]

    result = []
    for pair in pairs:
        theta, phi = pair
        cord = [sin(theta) * cos(phi) * r + mu[0],
                sin(theta) * sin(phi) * r + mu[1],
                cos(theta) * r + mu[2]]
        if cord in result:
            continue
        result.append(cord)

    return result


# #########################################################################
# 0. Base Loader
# #########################################################################
class BaseLoader(ABC):
    def __init__(self):
        super().__init__()
        self.train_set = None  # must be of type torch.utils.data.Dataset
        self.test_set = None  # must be of type torch.utils.data.Dataset

    @abstractmethod
    def loaders(self,
                batch_size: int,
                shuffle_train=True,
                shuffle_test=False,
                num_workers: int = 0) -> (DataLoader, DataLoader):
        """Implement data loaders of type torch.utils.data.DataLoader for train_set and test_set."""
        pass

    def __repr__(self):
        return self.__class__.__name__


# #########################################################################
# 1. Gaussian3D Dataset for Training
# #########################################################################
class Gaussian3DDataset(Dataset):
    def __init__(self,
                 normal_mu: str='1_-1_1',
                 abnormal_mu: str='',  # If unsupervised, do not specify
                 ratio_abnormal: float=0.1,
                 train: bool=True,
                 split: int=0.2,
                 random_state: int=42):
        super(Dataset, self).__init__()

        # Get the data for training and test
        X_train, X_test, y_train, y_test, _ = gen_gaussian_train(normal_mu,
                                                                 abnormal_mu,
                                                                 ratio_abnormal,
                                                                 split,
                                                                 random_state)

        # Transform to tensors
        if train:
            self.X = torch.tensor(X_train, dtype=torch.float32)
            self.y = torch.tensor(y_train, dtype=torch.float32)
        else:
            self.X = torch.tensor(X_test, dtype=torch.float32)
            self.y = torch.tensor(y_test, dtype=torch.float32)

    def __getitem__(self, index):
        sample, target = self.X[index], int(self.y[index])
        return sample, target, index

    def __len__(self):
        return len(self.X)


# #########################################################################
# 2. Gaussian3D Loader for Training
# #########################################################################
class Gaussian3DLoader(BaseLoader):
    def __init__(self,
                 normal_mu: str='1_-1_1',
                 abnormal_mu: str='',  # If unsupervised, do not specify
                 ratio_abnormal: float=0.1,
                 split: int=0.2,
                 random_state: int=42):
        super().__init__()

        # Get train set
        self.train_set = Gaussian3DDataset(normal_mu,
                                           abnormal_mu,
                                           ratio_abnormal,
                                           True,
                                           split,
                                           random_state)

        self.test_set = Gaussian3DDataset(normal_mu,
                                          abnormal_mu,
                                          ratio_abnormal,
                                          False,
                                          split,
                                          random_state)

    def loaders(self,
                batch_size: int=128,
                shuffle_train: bool=True,
                shuffle_test: bool=False,
                num_workers: int = 0):
        train_loader = DataLoader(dataset=self.train_set,
                                  batch_size=batch_size,
                                  shuffle=shuffle_train,
                                  num_workers=num_workers,
                                  drop_last=True)
        test_loader = DataLoader(dataset=self.test_set,
                                 batch_size=batch_size,
                                 shuffle=shuffle_test,
                                 num_workers=num_workers,
                                 drop_last=False)
        return train_loader, test_loader


# #########################################################################
# 3. Gaussian3D Dataset for Eval (Only load abnormal data!)
# #########################################################################
class Gaussian3DDatasetEval(Dataset):
    def __init__(self,
                 abnormal_mu_test,
                 normal_mu_train: str='1_-1_1',
                 abnormal_mu_train: str='',
                 ratio_abnormal: float=0.1,
                 split: int=0.2,
                 random_state: int=42):
        super(Dataset, self).__init__()

        np.random.seed(random_state)

        # Get the scaler used for training
        _, _, _, _, scaler = gen_gaussian_train(normal_mu_train,
                                                abnormal_mu_train,
                                                ratio_abnormal,
                                                split, random_state)

        # Generate abnormal X to test
        cov = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]
        X = np.random.multivariate_normal([int(i) for i in abnormal_mu_test.split('_')], cov, 6000 * ratio_abnormal)
        y = np.ones(X.shape[0])

        # Normalize the abnormal X
        X = scaler.transform(X)

        # Transform to tensors
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)


    def __getitem__(self, index):
        sample, target = self.X[index], int(self.y[index])
        return sample, target, index

    def __len__(self):
        return len(self.X)


# #########################################################################
# 4. Gaussian3D Loader for Eval
# #########################################################################
class Gaussian3DLoaderEval(BaseLoader):
    def __init__(self,
                 abnormal_mu_test,
                 normal_mu_train: str='1_-1_1',
                 abnormal_mu_train: str='',
                 ratio_abnormal: float=0.1,
                 split: int=0.2,
                 random_state: int=42):
        super().__init__()

        # Get train set
        self.all_set = Gaussian3DDatasetEval(abnormal_mu_test,
                                             normal_mu_train,
                                             abnormal_mu_train,
                                             ratio_abnormal,
                                             split,
                                             random_state)


    def loaders(self,
                batch_size: int=128,
                shuffle: bool=False,
                num_workers: int=0):
        all_loader = DataLoader(dataset=self.all_set,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                num_workers=num_workers,
                                drop_last=False)
        return all_loader
