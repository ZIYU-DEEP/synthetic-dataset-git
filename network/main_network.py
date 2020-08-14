"""
Title: main_network.py
Description: Build networks.
Reference: https://github.com/lukasruff/Deep-SAD-PyTorch/tree/master/src/networks
"""

from fmnist_LeNet import FashionMNISTLeNet, FashionMNISTLeNetAutoencoder
from kmnist_LeNet import KMNISTLeNet, KMNISTLeNetAutoencoder
from cifar10_LeNet import CIFAR10LeNet, CIFAR10LeNetAutoencoder
from gaussian3d_net import Guassian3DNet, Gaussian3DNetAutoencoder
from gaussian3d_net import *
from mlp import *


# #########################################################################
# 1. Build the Network Used for Training
# #########################################################################
def build_network(net_name='fmnist_LeNet_one_class'):
    # known_networks = ('fmnist_LeNet_one_class', 'fmnist_LeNet_rec',
    #                   'cifar10_LeNet_one_class', 'cifar10_LeNet_rec')
    # assert net_name in known_networks

    net_name = net_name.strip()

    # The network for the one-class model training
    if net_name == 'fmnist_LeNet_one_class':
        return FashionMNISTLeNet(rep_dim=64)

    # The network for the reconstruction model training
    if net_name == 'fmnist_LeNet_rec':
        return FashionMNISTLeNetAutoencoder(rep_dim=64)

    # The network for the one-class model training
    if net_name == 'kmnist_LeNet_one_class':
        return KMNISTLeNet(rep_dim=64)

    # The network for the reconstruction model training
    if net_name == 'kmnist_LeNet_rec':
        return KMNISTLeNetAutoencoder(rep_dim=64)

    # The network for the one-class model training
    if net_name == 'cifar10_LeNet_one_class':
        return CIFAR10LeNet(rep_dim=128)

    # The network for the reconstruction model training
    if net_name == 'cifar10_LeNet_rec':
        return CIFAR10LeNetAutoencoder(rep_dim=128)

    if net_name == 'gaussian3d_one_class':
        return Guassian3DNet(rep_dim=2)

    if net_name == 'gaussian3d_rec':
        return Gaussian3DNetAutoencoder(rep_dim=2)

    if net_name == 'satellite_mlp':
        return MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage_mlp':
        return MLP(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    return None


# #########################################################################
# 2. Build the Network Used for Pre-Training (Only for One-Class Model)
# #########################################################################
def build_autoencoder(net_name='fmnist_LeNet_one_class'):
    # known_networks = ('fmnist_LeNet_one_class', 'cifar10_LeNet_one_class')
    # assert net_name in known_networks

    net_name = net_name.strip()

    # The network for the one-class model pretraining
    if net_name == 'fmnist_LeNet_one_class':
        return FashionMNISTLeNetAutoencoder(rep_dim=64)

    # The network for the one-class model pretraining
    if net_name == 'kmnist_LeNet_one_class':
        return KMNISTLeNetAutoencoder(rep_dim=64)

    # The network for the one-class model pretraining
    if net_name == 'cifar10_LeNet_one_class':
        return CIFAR10LeNetAutoencoder(rep_dim=128)

    if net_name == 'gaussian3d_one_class':
        return Gaussian3DNetAutoencoder(rep_dim=2)

    if net_name == 'satellite_mlp':
        return MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    if net_name == 'satimage_mlp':
        return MLP_Autoencoder(x_dim=36, h_dims=[32, 16], rep_dim=8, bias=False)

    return None
