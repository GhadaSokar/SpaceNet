  
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

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def take_layer(self, name,param):
        if len(param.shape)>1:
            return True
        else:
            return False

    def last_layer(self,name):
        if ((name in self.layers_names[-1]) or (name in self.layers_names[-2])):
            return True
        else:
            return False

    def __init__(self):
        super(MLP, self).__init__()      
        #model
        self.fc1 = nn.Linear(28*28, 400, bias=True)
        nn.init.xavier_uniform(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        self.fc2 = nn.Linear(400, 400, bias=True)
        nn.init.xavier_uniform(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        self.fc3 = nn.Linear(400, 10, bias=True)
        nn.init.xavier_uniform(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

        self.no_params=[10000, 1640, 80]
        self.num_classes=10

        self.layers_names = []
        for name, param in self.named_parameters():
            if self.take_layer(name,param):
                self.layers_names.append(name)
        self.layers_names.append(name)

    def forward(self, x):
        x0 = x.view(-1, 28*28)
        x1 = F.relu(self.fc1(x0))
        x2 = F.relu(self.fc2(x1))
        x3 = self.fc3(x2)
        return x3


