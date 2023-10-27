from torchvision import models
import torch.nn as nn
import torch
import timm
from MedViT import MedViT_small
from MedViT import MedViT_base
from MedViT import MedViT_large

class EnsembleModel(nn.Module):
    def __init__(self, base_models):
        super(EnsembleModel, self).__init__()
        self.base_models = nn.ModuleList(base_models)

    def forward(self, x):
        predictions = []
        for model in self.base_models:
            model.eval()
            with torch.no_grad():
                output = model(x)
            predictions.append(output)

        # Combine the predictions (averaging in this case)
        ensemble_output = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_output

resnet_list = ['resnet18', 'resnet101', 'resnet152']
resnet_mod_list = ['resnet18_mod1', 'resnet101_mod1', 'resnet152_mod1']
densenet_list = ['densenet121', 'densenet161', 'densenet169', 'densenet201']
vit_list = ['vit_small', 'vit_base', 'vit_large']
medvit_list = ['medvit_small', 'medvit_base', 'medvit_large']

def get_model_structure(model_name, pretrain=None):
    if model_name == 'resnet18' or model_name == 'resnet18_mod1':
        return models.resnet18(weights=pretrain)
    elif model_name == 'resnet101' or model_name == 'resnet101_mod1':
        return models.resnet101(weights=pretrain)
    elif model_name == 'resnet152' or model_name == 'resnet152_mod1':
        return models.resnet152(weights=pretrain)
        
    elif model_name == 'densenet121':
        return models.densenet121(weights=pretrain)
    elif model_name == 'densenet161':
        return models.densenet161(weights=pretrain)
    elif model_name == 'densenet169':
        return models.densenet169(weights=pretrain)
    elif model_name == 'densenet201':
        return models.densenet201(weights=pretrain)
        
    elif model_name == 'vit_large':
        return timm.create_model("vit_large_patch16_224", pretrained=pretrain)
    
    elif model_name == "medvit_large":
        return MedViT_large(pretrained = pretrain)
    elif model_name == "medvit_base":
        return MedViT_base(pretrained = pretrain)
    elif model_name == "medvit_small":
        return MedViT_small(pretrained = pretrain)
    return None

def get_ensemble(model_name, pretrain, class_counts, pretrain_category, dropout_prob):
    model_list = []
    for model in model_name:
        temp_model = get_model(model_name, pretrain, class_counts, pretrain_category, dropout_prob)
        model_list.append(temp_model)
    model = EnsembleModel(model_list)
    return model

def get_model(model_name, pretrain, class_counts, pretrain_category, dropout_prob):
    # This is for building ensemble model 
    if isinstance(model_name, list):
        model = get_ensemble(model_name, pretrain, class_counts, pretrain_category, dropout_prob)
        print(f"Ensemble model load sucessfully!")

    # Using my own pretrained model weights
    elif isinstance(pretrain, str): # using my own pretrained weight.
        print(f"Loading up your own model weight:{pretrain}")
        model = get_model_structure(model_name)
        if model_name in resnet_list:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, pretrain_category)


        print(f"Weight Loaded up successfully!")

    # Building up model structure
    else:
        if isinstance(pretrain, str): # using my own pretrained weight.
            print(f"Loading up your own model weight:{pretrain}")
            model = get_model_structure(model_name, False)
        else: 
            model = get_model_structure(model_name, pretrain)
            
        if model_name in resnet_mod_list:
            print("## YOU ARE USING A MODED MODEL ##")
            num_ftrs = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(p=dropout_prob),
                nn.Linear(num_ftrs, len(class_counts)),
            )
        elif model_name in resnet_list:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, len(class_counts))
            
        elif model_name in densenet_list:
            num_ftrs = model.classifier.in_features
            model.classifier = nn.Linear(num_ftrs, len(class_counts))
            
        elif model_name in vit_list:
            num_ftrs = model.head.in_features
            model.head = nn.Linear(num_ftrs, len(class_counts))
            
        elif model_name in medvit_list:
            model.proj_head[0] = torch.nn.Linear(in_features=1024, out_features=len(class_counts), bias=True)

        if isinstance(pretrain, str): # using my own pretrained weight.
            state_dict = torch.load(pretrain)
            model.load_state_dict(state_dict)

    return model