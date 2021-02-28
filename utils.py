  
# Author: Ghada Sokar et al.
# This is the implementation for the SpaceNet: Make Free Space for Continual Learning paper in NeuroComputing Journal
# if you use part of this code, please cite the following article:
# @article{SOKAR20211,
# title = {SpaceNet: Make Free Space for Continual Learning},
# journal = {Neurocomputing},
# volume = {439},
# pages = {1-11},
# year = {2021},
# issn = {0925-2312},
# doi = {https://doi.org/10.1016/j.neucom.2021.01.078},
# url = {https://www.sciencedirect.com/science/article/pii/S0925231221001545},
# author = {Ghada Sokar and Decebal Constantin Mocanu and Mykola Pechenizkiy}
# }
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import copy

def get_task_load_train(train_dataset,batch_size):
    train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size,
    num_workers=0,
    pin_memory=True, shuffle=True)
    print('Train loader length', len(train_loader))
    return train_loader

def get_task_load_test(test_dataset,test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True)
    return test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
    #transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('../data', train=False, transform=transform)
    return full_dataset,test_dataset

def task_construction(task_labels):
    full_dataset,test_dataset=load_data()
    train_dataset=split_dataset_by_labels(full_dataset, task_labels)
    test_dataset=split_dataset_by_labels(test_dataset, task_labels)
    return train_dataset,test_dataset

def split_dataset_by_labels(dataset, task_labels):
    datasets = []
    for labels in task_labels:
        idx=np.in1d(dataset.targets, labels)
        splited_dataset=copy.deepcopy(dataset)
        splited_dataset.targets = splited_dataset.targets[idx]
        splited_dataset.data = splited_dataset.data[idx]
        datasets.append(splited_dataset)
    return datasets
