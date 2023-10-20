from C_BASELINE import train_mod
import torch
from C_other_func import Notification
name = "diff_dataset_1020"
path = "D:/REDO/RESNET152/"
dropout_prob = 0
ii= 1
model_things = {
    'data_dir' : "D:/P2023/DATA/glomer_cg",
    'train_ratio' : 0.6,
    'val_ratio' : 0.5,
    'random_seed' : 42,
    'batch_size' : 40,
    'log_path' : f"{path}{name}.txt",
    'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
    'learning_rate' : 0.01,
    'num_of_epoch' : 20,
    'lr_method' : "LR_stepping",
    'pretrain' : True,
    'pretrain_category' : None,
    'model_name' : 'resnet152',
    'other_info' : "Try to test model and see the result of classifying in different dataset. \nHope it goes well",
    'data_transforms_op' : 2,
    'dropout_prob' :  dropout_prob
}
model = train_mod(model_things)
weight_store_path = model_things['weight_store_path']
torch.save(model.state_dict(), weight_store_path)

dropout_prob = 0
ii= 2
model_things = {
    'data_dir' : "D:/P2023/DATA/glomer_tvgh",
    'train_ratio' : 0.6,
    'val_ratio' : 0.5,
    'random_seed' : 42,
    'batch_size' : 40,
    'log_path' : f"{path}{name}.txt",
    'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
    'learning_rate' : 0.01,
    'num_of_epoch' : 20,
    'lr_method' : "LR_stepping",
    'pretrain' : True,
    'pretrain_category' : None,
    'model_name' : 'resnet152',
    'other_info' : "Try to test model and see the result of classifying in different dataset. \nHope it goes well",
    'data_transforms_op' : 2,
    'dropout_prob' :  dropout_prob
}
model = train_mod(model_things)
weight_store_path = model_things['weight_store_path']
torch.save(model.state_dict(), weight_store_path)

# model 1
name = "diff_dataset_1020"
path = "D:/REDO/RESNET101/"
dropout_prob = 0
ii= 1
model_things = {
    'data_dir' : "D:/P2023/DATA/glomer_cg",
    'train_ratio' : 0.6,
    'val_ratio' : 0.5,
    'random_seed' : 42,
    'batch_size' : 40,
    'log_path' : f"{path}{name}.txt",
    'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
    'learning_rate' : 0.01,
    'num_of_epoch' : 20,
    'lr_method' : "LR_stepping",
    'pretrain' : True,
    'pretrain_category' : None,
    'model_name' : 'resnet101',
    'other_info' : "Try to test model and see the result of classifying in different dataset. \nHope it goes well",
    'data_transforms_op' : 2,
    'dropout_prob' :  dropout_prob
}
model = train_mod(model_things)
weight_store_path = model_things['weight_store_path']
torch.save(model.state_dict(), weight_store_path)

dropout_prob = 0
ii= 2
model_things = {
    'data_dir' : "D:/P2023/DATA/glomer_tvgh",
    'train_ratio' : 0.6,
    'val_ratio' : 0.5,
    'random_seed' : 42,
    'batch_size' : 40,
    'log_path' : f"{path}{name}.txt",
    'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
    'learning_rate' : 0.01,
    'num_of_epoch' : 20,
    'lr_method' : "LR_stepping",
    'pretrain' : True,
    'pretrain_category' : None,
    'model_name' : 'resnet101',
    'other_info' : "Try to test model and see the result of classifying in different dataset. \nHope it goes well",
    'data_transforms_op' : 2,
    'dropout_prob' :  dropout_prob
}
model = train_mod(model_things)
weight_store_path = model_things['weight_store_path']
torch.save(model.state_dict(), weight_store_path)


