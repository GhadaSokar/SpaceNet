  
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
import torch.optim as optim
import argparse
import sys
import numpy as np
import os
import random
import time

import models
import CL
import utils

def test_previous_tasks(args, model, criterion, device,current_task_idx,test_dataset):
    average=0
    cnt=0    
    for i in range(current_task_idx+1):
        print('Task '+str(i))
        test_loader=utils.get_task_load_test(test_dataset[i],args.batch_size)
        val_acc = evaluate(args,i, model, criterion, device, test_loader, is_test_set=True)
        average+=val_acc
        cnt+=1
    print(f"average acc over {cnt} tasks = {average/cnt}")

def evaluate(args,current_task ,model, criterion, device, test_loader, is_test_set=False):
    test_loss = 0
    correct = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            n += target.shape[0]
    test_loss /= float(n)

    print('\n{}: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        'Test evaluation' if is_test_set else 'Evaluation',
        test_loss, correct, n, 100. * correct / float(n)))
    sys.stdout.flush()                
    return correct / float(n)
        
def main():  
    parser = argparse.ArgumentParser(description='CL on Split MNIST benchmark using SpaceNet')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0,
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--l2', type=float, default=0)
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=13, help='random seed (default: 13)')
    parser.add_argument('--log_interval', type=int, default=50,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save_model_path', type=str, default='./models/model.pt', help='path for saving the final model')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.environ['PYTHONHASHSEED']=str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    task_labels = [[0,1],[2,3],[4,5],[6,7],[8,9]]
    num_tasks=len(task_labels)

    specific_nodes_count = [0,40,40]
    selected_nodes_count = 80

    # read data
    train_dataset,test_dataset=utils.task_construction(task_labels)
    # model
    model = models.MLP().to(device)
    CL_obj=CL.CL(model, device, specific_nodes_count, selected_nodes_count, task_labels)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(CL_obj.model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.l2, nesterov=False)

    CL_obj.set_init_network_weight()
    for task_idx in range(0,num_tasks):
        # get current task data
        train_loader = utils.get_task_load_train(train_dataset[task_idx],args.batch_size)
        test_loader = utils.get_task_load_test(test_dataset[task_idx],args.batch_size)
        # reset and neuron importance for the current task
        CL_obj.reset_importance()

        for epoch in range(args.epochs):
            CL_obj.model.train()
            t0=time.time()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device) 
                CL_obj.save_old_tasks_weights()
                optimizer.zero_grad()
                outputs = CL_obj.model(data)
                loss = criterion(outputs, target) 
                loss.backward()
                CL_obj.apply_mask_on_grad()
                optimizer.step()
                CL_obj.calculate_importance()
                CL_obj.recover_old_tasks_weights()

                if batch_idx % args.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader)*args.batch_size,
                    100. * batch_idx / len(train_loader), loss.item()))
                    sys.stdout.flush()    

            print('Current learning rate: {0}. Time taken for epoch: {1:.2f} seconds.\n'.format(optimizer.param_groups[0]['lr'], time.time() - t0))
            
            # perform Drop and Grow phases
            if epoch < args.epochs-1:
                CL_obj.drop()
                CL_obj.grow()

        val_acc = evaluate(args,task_idx, CL_obj.model, criterion, device, test_loader, is_test_set=True)
       
        if task_idx<num_tasks-1:
            CL_obj.prepare_next_task()

    # print(len(list(set(CL_obj.used_neurons_layer_1))))
    # print(len(list(set(CL_obj.used_neurons_layer_2))))

    CL_obj.set_classifer_to_all_learned_tasks()
    test_previous_tasks(args, CL_obj.model, criterion, device, task_idx, test_dataset)
    torch.save(CL_obj.model, args.save_model_path)
if __name__ == '__main__':
   main()