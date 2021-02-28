# SpaceNet: Make Free Space For Continual Learning
This is the official PyTorch implementation for the [SpaceNet: Make Free Space For Continual Learning](https://www.sciencedirect.com/science/article/pii/S0925231221001545?via%3Dihub) paper in Elsevier Neurocomputing Journal. 

# Abstract
In this work, we present a new brain-inspired method to learn a sequence of tasks sequentially. SpaceNet learns semi-distributed sparse representation to mitigate forgetting previously learned knowledge. We train each task using sparse connections from scratch. Our proposed adaptive sparse training algorithm dynamically redistributes and compacts the sparse connections in the important neurons to the current task leaving free space for future tasks.

# Requirements
* Python 3.6
* Pytorch 1.2
* torchvision 0.4

# Usage
You can use main.py to run SpaceNet on the Split MNIST benchmark. 

```
python main.py
```

# Reference
If you use this code, please cite our paper:
```
@article{SOKAR20211,
title = {SpaceNet: Make Free Space for Continual Learning},
journal = {Neurocomputing},
volume = {439},
pages = {1-11},
year = {2021},
issn = {0925-2312},
doi = {https://doi.org/10.1016/j.neucom.2021.01.078},
url = {https://www.sciencedirect.com/science/article/pii/S0925231221001545},
author = {Ghada Sokar and Decebal Constantin Mocanu and Mykola Pechenizkiy}
}
```
