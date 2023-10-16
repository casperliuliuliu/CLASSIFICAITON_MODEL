from BASELINE import train_mod
from torchvision.models import ResNet18_Weights
import torch
# setup 1/2
name = "ResNet18_1016_pyfile"
model_things = {
    'data_dir' : "D:/P2023/DATA/glomer_cg(2)",
    'train_ratio' : 0.6,
    'val_ratio' : 0.5,
    'random_seed' : 42,
    'batch_size' : 4,
    'log_path' : f"D:/REDO/RESNET18/{name}.txt",
    'weight_store_path' : f"D:/REDO/RESNET18/WEIGHT/{name}(1).pt",
    'learning_rate' : 0.01,
    'num_of_epoch' : 1,
    'lr_method' : "LR_stepping",
    'pretrain' : ResNet18_Weights.IMAGENET1K_V1,
    'model_name' : 'resnet18'
}
test_mod(model_things)
