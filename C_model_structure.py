from torchvision import datasets, models, transforms
import torch.nn as nn
import torch

def get_model_structure(model_name, pretrain=None):
    if model_name == 'resnet18':
        return models.resnet18(weights=pretrain)
    elif model_name == 'resnet101':
        return models.resnet101(weights=pretrain)
    elif model_name == 'resnet152':
        return models.resnet152(weights=pretrain)
    return None
    
def get_model(model_name, pretrain, class_counts, pretrain_category):
    if isinstance(pretrain, str): # using my own pretrained weight.
        print(f"Loading up your own model weight:{pretrain}")
        model = get_model_structure(model_name)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, pretrain_category)
        state_dict = torch.load(pretrain)
        model.load_state_dict(state_dict)
        print(f"Weight Loaded up successfully!")
        
    else:
        model = get_model_structure(model_name, pretrain)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, len(class_counts))

    return model