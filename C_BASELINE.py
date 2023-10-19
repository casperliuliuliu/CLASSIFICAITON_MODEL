# Package
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import random_split
from tqdm import tqdm

from torch.utils.data import DataLoader, Subset
from datetime import datetime
import random
from torch.utils.data import ConcatDataset
from C_model_structure import get_model
from C_datatransform import get_data_transforms
from C_other_func import write_log, send_email
filename = ""
    
def get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size):
    # Create a single merged dataset
    train_dataset = datasets.ImageFolder(data_dir, transform = data_transforms['train'])
    val_dataset = datasets.ImageFolder(data_dir, transform = data_transforms['val'])
    test_dataset = datasets.ImageFolder(data_dir, transform = data_transforms['test'])
    # if 'aug0' in data_transforms.keys():
    #     merge_dataset = train_dataset
    # else:
    
    # obtain training indices that will be used for validation
    num_train = len(test_dataset)
    indices = list(range(num_train))
    random.shuffle(indices)
    split_train = int(np.floor(train_ratio * num_train))
    split_val = split_train + int(np.floor(val_ratio * (num_train-split_train)))
    train_idx, val_idx, test_idx = indices[0:split_train], indices[split_train:split_val], indices[split_val:]
    merge_dataset = Subset(train_dataset, train_idx)
    print(data_transforms.keys())
    for ii in range(len(data_transforms.keys())-3):
        print(ii)
        aug_dataset = datasets.ImageFolder(data_dir, transform = data_transforms[f'aug{ii}'])
        aug_sub = Subset(aug_dataset, train_idx)
        merge_dataset = ConcatDataset([merge_dataset,aug_sub])
    
    train_loader = DataLoader(merge_dataset, batch_size=batch_size)
    val_loader = DataLoader(Subset(val_dataset, val_idx), batch_size=batch_size)
    test_loader = DataLoader(Subset(test_dataset, test_idx), batch_size=batch_size)
    
    # check dataset
    pprint(f"Total number of samples: {num_train} datapoints")
    pprint(f"Number of train samples: {len(train_loader)} batches/ {len(train_loader.dataset)} datapoints")
    pprint(f"Number of val samples: {len(val_loader)} batches/ {len(val_loader.dataset)} datapoints")
    pprint(f"Number of test samples: {len(test_loader)} batches/ {len(test_loader.dataset)} datapoints")
    pprint(f"")
    
    dataloaders = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader,
    }
    return dataloaders
    
def get_class_counts(data_dir):
    train_dataset = datasets.ImageFolder(data_dir)
    class_counts = {}
    for _, label in train_dataset.samples:
        class_name = train_dataset.classes[label]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    return class_counts
    
def pprint(output = '\n', show_time = False): # print and fprint at the same time
    global filename
    print(output)
    with open(filename, 'a') as f:
        if show_time:
            f.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))

        f.write(str(output))
        f.write('\n')

def get_dataset_sizes(dataloaders):
    dataset_sizes = {
        'train': len(dataloaders['train'].dataset),
        'val': len(dataloaders['val'].dataset),
        'test': len(dataloaders['test'].dataset)
    }
    return dataset_sizes
    
def train_model(model, model_things):
    NUM_EPOCHS = model_things['num_of_epoch']
    data_dir = model_things['data_dir']
    learning_rate = model_things['learning_rate']
    train_ratio = model_things['train_ratio']
    val_ratio = model_things['val_ratio']
    batch_size = model_things['batch_size']
    # model_name = model_things['model_name']
    data_transforms_op = model_things['data_transforms_op']
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    class_counts = get_class_counts(data_dir)
    data_transforms = get_data_transforms(data_transforms_op)
    dataloaders = get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size)
    dataset_sizes = get_dataset_sizes(dataloaders)
    log_message = write_log(model_things,class_counts)
    
    model = model.cuda()
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        pprint('Epoch [%d/%d]'% (epoch+1, NUM_EPOCHS), show_time=True)
        pprint('-' * 10)
        pprint("Learning rate:{}".format(optimizer.param_groups[0]['lr']))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            num_class = len(class_counts)
            confus = torch.zeros(num_class, num_class,dtype=int)            
            for inputs, labels in tqdm(dataloaders[phase]): # Iterate over data.
                inputs, labels = inputs.cuda(), labels.cuda()
                with torch.set_grad_enabled(phase == 'train'): # forward # track history if only in train
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    # print(loss)
                    if phase == 'train': # backward + optimize only if in training phase
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                for ii in range(len(preds)):# statistics
                    confus[ labels.data[ii] ][ preds[ii] ]+=1
                    
            if phase == 'train':
                step_lr_scheduler.step()
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            pprint(confus)
            pprint('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict()) 
        print()
    time_elapsed = time.time() - since
    pprint('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    pprint('Best val Acc: {:.4f}'.format(
                best_acc))
    log_message += '\n  Whole training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    log_message +='\n Best val Acc={:.4f}'.format(
                best_acc)
    
    send_email(log_message, model_name)
    
    pprint()
    pprint()
    model.load_state_dict(best_model_wts) # load best model weights
    return model
    
def train_mod(model_things):
    log_path = model_things['log_path']
    pretrain = model_things['pretrain']
    model_name = model_things['model_name']
    data_dir = model_things['data_dir']
    pretrain_category = model_things['pretrain_category']
    dropout_prob = model_things['dropout_prob']
    
    global filename
    filename = log_path
    
    class_counts = get_class_counts(data_dir)
    model = get_model(model_name, pretrain, class_counts, pretrain_category, dropout_prob)
    print(model)
    model = train_model(model, model_things)
    return model
    