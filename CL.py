  
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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import copy
import models

class CL():    
    def __init__(self, model, device, specific_nodes_count, selected_nodes_count, task_labels):
        self.model=model
        self.device=device        
        self.replace_percentage = 0.2
        self.inf = 99999
        self.mask = {}
        self.previous_mask= {}
        self.task_labels = task_labels
        self.current_task=0
        self.specific_nodes_count_per_layer=specific_nodes_count
        self.selected_nodes_count=selected_nodes_count
        self.used_neurons_layer_1 = []
        self.used_neurons_layer_2 = []
        self.init_free_nodes()
        self.init_prev_masks()
        self.create_masks()


    def init_free_nodes(self):
        self.layers_free_nodes = {}
        self.num_specific_nodes = {}
        i=0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.layers_free_nodes[name] = torch.ones(
                        param.shape[1]).to(self.device)

                    self.num_specific_nodes[name] = self.specific_nodes_count_per_layer[i]
                    i += 1

        # free nodes for the output layer 
        self.layers_free_nodes[name] = torch.zeros(
                        self.model.num_classes).to(self.device)
        self.layers_free_nodes[name][self.task_labels[self.current_task]] = 1                
        self.last_layer_active_task = torch.zeros(
                        self.model.num_classes).to(self.device)
        self.last_layer_active_task[self.task_labels[self.current_task]] = 1 

    # neurons reservation
    def update_free_nodes(self):
        idx=0
        # remove specific neurons of the current task from free list
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    if idx>0:
                        layeridxImp_tmp = np.argsort(self.layers_importnace[name].numpy())[::-1]
                        tmp_layer = copy.copy(self.layers_free_nodes[name].numpy())
                        tmp_layer[layeridxImp_tmp[:self.num_specific_nodes[name]]] = 0
                        self.layers_free_nodes[name] = torch.from_numpy(tmp_layer)
                    idx+=1

        # switch active layer to next task
        self.layers_free_nodes[name]=torch.zeros(
                        self.model.num_classes).to(self.device)
        self.layers_free_nodes[name][self.task_labels[self.current_task+1]] = 1
        self.last_layer_active_task=torch.zeros(
                        self.model.num_classes).to(self.device)
        self.last_layer_active_task[self.task_labels[self.current_task+1]] = 1

    def prepare_next_task(self):
        # add mask of the current task to the previous tasks masks  
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.previous_mask[name] += self.mask[name]
                    self.previous_mask[name][self.previous_mask[name]>1] = 1
    
        # neurons reservation 
        self.update_free_nodes()
        self.current_task += 1  
        # connections allocation      
        self.create_masks()
        self.retain_last_layer_and_init_next_task_weights()

    def retain_last_layer_and_init_next_task_weights(self):
        for name, param in self.model.named_parameters():
            if self.model.take_layer(name,param):
                # Retain the connections of last layer for task t
                if self.model.last_layer(name):
                    # save new learned weights for the last layer in init_weights
                    self.init_weights[name][self.task_labels[self.current_task-1],:] = param.data[self.task_labels[self.current_task-1],:]
                    param.data=torch.zeros_like(self.init_weights[name])

                # random init new weights for the next task
                param.data[self.mask[name]==1] = self.init_weights[name][self.mask[name]==1]

    # return all the retain classifiers for the tasks seen so far
    def set_classifer_to_all_learned_tasks(self):
        for name, param in self.model.named_parameters():
            if self.model.take_layer(name,param):
                if self.model.last_layer(name):
                    for i in range(self.current_task):
                        param.data[self.task_labels[i],:] = self.init_weights[name][self.task_labels[i],:] 


    def reduce(self, tensor):
        if(len(tensor.shape) == 2):
            return tensor
        return tensor.sum(dim=(2, 3))

    def set_init_network_weight(self):
        self.init_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):                   
                    self.init_weights[name] = copy.deepcopy(param.data)
                    param.data = param.data*self.mask[name].to(self.device)

    def init_prev_masks(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):                    
                    self.previous_mask[name] = torch.zeros_like(
                        param.data).to(self.device)    
                    print(name, param.data.shape)

    def create_masks(self):
        idx=0
        self.selected_nodes={}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):  
                    temp_mask = copy.deepcopy(self.previous_mask[name])
                    self.mask[name]=torch.zeros_like(self.previous_mask[name])
                    temp_mask=self.reduce(temp_mask)
                    #  masking all nodes except the selected 
                    #  for layer i the selected nodes are the selected nodes for layer j of previous layer
                    if idx>0:
                        nodes_layer_i=copy.deepcopy(nodes_layer_j)
                        temp_mask[:,nodes_layer_i] =1
                    
                    if not self.model.last_layer(name):
                        # select nodes layer j
                        # get idnicies of free nodes
                        Free_idx_next_layer=torch.where(self.layers_free_nodes[self.model.layers_names[idx+1]]==1)
                        # mask unselected nodes 
                        nodes_layer_j=np.random.choice(Free_idx_next_layer[0].numpy(), size=Free_idx_next_layer[0].shape[0]-self.selected_nodes_count, replace=False)
                        temp_mask[nodes_layer_j,:] = 1
                        # mask specific nodes for previous tasks in layer i,j
                        temp_mask[:,self.layers_free_nodes[self.model.layers_names[idx]]==0] =1 
                        temp_mask[self.layers_free_nodes[self.model.layers_names[idx+1]]==0,:] = 1 
                    else:
                        temp_mask[:,self.layers_free_nodes[self.model.layers_names[idx]]==0] = 1 
                        temp_mask[self.last_layer_active_task==0,:]=1
                        self.selected_nodes[self.model.layers_names[idx+1]]=torch.where(self.last_layer_active_task==1)
                    
                    # the remaining elements is temp_mask is the places where we can allocate connection for the current task
                    idx_zeros_i,idx_zeros_j=np.where(temp_mask == 0)
                    self.selected_nodes[self.model.layers_names[idx]]=list(set(idx_zeros_j))
                    
                    ## for debugging and statistics
                    #if idx!=0:
                    #    print(self.selected_nodes[self.model.layers_names[idx]])
                    if idx==1:
                        self.used_neurons_layer_1 = self.used_neurons_layer_1 + list(set(idx_zeros_j))
                    elif idx==2:
                        self.used_neurons_layer_2 = self.used_neurons_layer_2 + list(set(idx_zeros_j))   
                    
                    new_conn_idx = np.random.choice(range(idx_zeros_i.shape[0]), size=int(self.model.no_params[idx]),replace=False)
                    if len(self.mask[name].shape)>2:
                        self.mask[name][idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx],:,:] = 1
                    else: 
                        self.mask[name][idx_zeros_i[new_conn_idx],idx_zeros_j[new_conn_idx]] = 1

                    idx+=1   

    def save_old_tasks_weights(self):
        self.old_weights = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.old_weights[name] = copy.deepcopy(param.data)

    def recover_old_tasks_weights(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param) and not self.model.last_layer(name):
                    param.data[self.previous_mask[name]==1]  = self.old_weights[name][self.previous_mask[name]==1] 

    def apply_mask_on_grad(self):
        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param): 
                    param.grad = param.grad*self.mask[name].to(self.device)
                    idx+=1
                elif 'bias' in name:
                    param.grad = param.grad*self.layers_free_nodes[self.model.layers_names[idx]].to(self.device)

    
    def drop(self):
        self.removed_mask = {}
        self.replace_count = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param) and not self.model.last_layer(name):
                    reduced_mask=self.reduce(self.mask[name])
                    reduced_mask[reduced_mask>0] = 1
                    total = torch.sum(reduced_mask)
                    replace_count = int(total*self.replace_percentage)
                    self.replace_count[name] = replace_count
                    importance = copy.deepcopy(
                        self.weights_importance[name]).to(self.device)
                    # importance = copy.deepcopy(
                    #     param.data).to(self.device)
                    importance += ((1-self.mask[name])*self.inf)
                    reduced_importance=self.reduce(abs(importance))
                    reduced_importance = reduced_importance.flatten()
                    #print(reduced_importance.shape)
                    #print(replace_count) 
                    idx = np.argpartition(reduced_importance.to("cpu"), replace_count)
                    removed_mask = torch.zeros_like(reduced_importance).to(self.device)
                    removed_mask[idx[:replace_count]] = 1
                    removed_mask = removed_mask.reshape(
                        reduced_mask.shape)
                    self.removed_mask[name] = copy.deepcopy(
                        removed_mask).to(self.device)

    def grow(self):
        for idx in range(len(self.model.layers_names)-2):
            name = self.model.layers_names[idx]
            nxt_name = self.model.layers_names[idx+1]
            layer_importnace = torch.mm(self.layers_importnace[nxt_name].reshape(self.layers_importnace[nxt_name].shape[0], 1),
                                        self.layers_importnace[name].reshape(self.layers_importnace[name].shape[0], 1).T)
            not_selected_nodes=torch.ones_like(layer_importnace)
            not_selected_nodes[:,self.selected_nodes[name]]-= 1
            not_selected_nodes[self.selected_nodes[nxt_name],:]-= 1
            not_selected_nodes[not_selected_nodes==0]=1
            not_selected_nodes[not_selected_nodes==-1]=0
            
            reduced_mask=self.reduce(self.mask[name]+self.previous_mask[name])
            reduced_mask[reduced_mask>0] = 1
            layer_importnace[reduced_mask==1] = -self.inf
            layer_importnace[not_selected_nodes==1] = -self.inf
            layer_importnace = -layer_importnace.flatten()
            idx_add = np.argpartition(layer_importnace.to(
                "cpu"), self.replace_count[name])
            
            assert(torch.max(layer_importnace[idx_add[:self.replace_count[name]]])<self.inf) 

            added_mask = torch.zeros_like(layer_importnace).to(self.device)
            added_mask[idx_add[:self.replace_count[name]]] = 1
            added_mask = added_mask.reshape(
                reduced_mask.shape)

            self.mask[name][self.removed_mask[name]==1]=0
            self.mask[name][added_mask==1]=1
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param) and not self.model.last_layer(name):
                    param.data = param.data*(self.mask[name]+self.previous_mask[name]).to(self.device)
        
    def reset_importance(self):
        self.weights_importance = {}
        self.layers_importnace = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.weights_importance[name] = torch.zeros_like(
                       param.data)
                    self.layers_importnace[name] = torch.zeros(
                        param.shape[1]).to(self.device)

    def calculate_importance(self):
        idx = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if self.model.take_layer(name,param):
                    self.weights_importance[name] += abs(
                        (param.data-self.old_weights[name])*param.grad*self.mask[name])
                    if idx == 0: # for input layer the importance of each node is based on the importance of the outlinks
                        layer_importnace = torch.sum(
                            self.weights_importance[name], dim=0).squeeze()                    
                    else:
                        layer_importnace = torch.sum(
                            self.weights_importance[self.model.layers_names[idx-1]], dim=1).squeeze()
                    idx+=1
                    if(len(layer_importnace.shape) > 1):
                        layer_importnace = layer_importnace.sum(
                            dim=(-1, -2)).squeeze().to(self.device)
                    self.layers_importnace[name] += layer_importnace
                    self.layers_importnace[name] *= self.layers_free_nodes[name]


    

