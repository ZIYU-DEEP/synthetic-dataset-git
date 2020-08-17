from abc import ABC, abstractmethod
from torch.utils.data import Subset
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from sklearn.model_selection import train_test_split

import torch
import scipy.io
import numpy as np


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
# 1. Dataset for Training
# #########################################################################
class SatelliteDataset(Dataset):
    def __init__(self,
                 root: str='../data/satellite.mat',
                 label_abnormal: tuple=(),  # If unsupervised, do not specify
                 train: bool=True,
                 split: float=0.2,
                 random_state: int=42):
        super(Dataset, self).__init__()

        # Initialization
        self.root = root
        self.label_abnormal = label_abnormal

        # Load data
        mat = scipy.io.loadmat(root)

        X = mat['X']
        y = mat['y'].reshape(-1)

        if not label_abnormal:
            X = X[y == 0]
            y = y[y == 0]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=split,
                                                            random_state=random_state,
                                                            stratify=y)

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
# 2. Loader for Training
# #########################################################################
class SatelliteLoader(BaseLoader):
    def __init__(self,
                 root: str='../data/satellite.mat',
                 label_abnormal: tuple=(),  # If unsupervised, do not specify
                 split: float=0.2,
                 random_state: int=42):
        super().__init__()

        # Get train set
        self.train_set = SatimageDataset(root,
                                         label_abnormal,
                                         True,
                                         split,
                                         random_state)

        self.test_set = SatimageDataset(root,
                                        label_abnormal,
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
