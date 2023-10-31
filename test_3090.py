from C_testing import eval_model
import torch
from C_other_func import Notification


"""
glomer_cg:
    "D:\P2023\DATA\glomer_cg(2)" -> 2 classes.
    "D:\P2023\DATA\glomer_cg" -> 5 classes.

glomer_cg_2020_2022:
    "D:\P2023\DATA\glomer_cg_2020_2022" -> 5 classes.

glomer_tvgh:
    "D:\P2023\DATA\glomer_tvgh" -> 2 classes.

"""

dataset = ["D:\P2023\DATA\glomer_cg(2)", "D:\P2023\DATA\glomer_cg", "D:\P2023\DATA\glomer_2020_2022", "D:\P2023\DATA\glomer_tbgh"]


# model_list = ['resnet101']
# model_weight = ['D:/REDO/resnet101/WEIGHT/diff_dataset_1020(3).pt']
# pretrain_category = [2]
# for ii in range(len(model_list)):
#     mod_running = model_list[ii]
#     print(mod_running)
#     name = "first_pretrain_test_1031"
#     path = f"D:/REDO/{mod_running}/"
#     dropout_prob = None
#     model_things = {
#         'data_dir' : dataset[1],
#         'train_ratio' : 0.6,
#         'val_ratio' : 0.5,
#         'random_seed' : 42,
#         'batch_size' : 5,
#         'log_path' : f"{path}{name}.txt",
#         'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
#         'learning_rate' : 0.01,
#         'num_of_epoch' : 20,
#         'lr_method' : "LR_stepping",
#         'pretrain' : model_weight[ii],
#         'pretrain_category' : pretrain_category[ii],
#         'model_name' : mod_running,
#         'other_info' : "To train for ensemble base model weight.",
#         'data_transforms_op' : 2,
#         'dropout_prob' :  dropout_prob,
#         'ensemble_model': False,
#         'eval_dataset': 'val'
#     }
#     model = eval_model(model_things)

# model_list = ['resnet101', 'resnet152', 'resnet101']
# pretrain = ['D:/REDO/resnet101/WEIGHT/diff_dataset_1020(2).pt', 'D:/REDO/resnet152/WEIGHT/diff_dataset_1020(0).pt', True]
# pretrain_category = [5, 5, None]
# dropout_prob = [None, None, None]

# model_list = ['resnet101']
# pretrain = ['D:/REDO/resnet101/WEIGHT/diff_dataset_1020(2).pt']
# pretrain_category = [5]
# dropout_prob = [None]

model_list = ['medvit_large']
pretrain = [True]
pretrain_category = [None]
dropout_prob = [None]

mod_running = "ensemble_resnet2"
print(mod_running)

name = "ensemble_test_1031"
path = f"D:/REDO/{mod_running}/"

model_things = {
    'data_dir' : dataset[1],
    'train_ratio' : 0.6,
    'val_ratio' : 0.5,
    'random_seed' : 42,
    'batch_size' : 5,
    'log_path' : f"{path}{name}.txt",
    'weight_store_path' : f"{path}/WEIGHT/{name}({0}).pt",
    'learning_rate' : 0.01,
    'num_of_epoch' : 20,
    'lr_method' : "LR_stepping",
    'pretrain' : pretrain,
    'pretrain_category' : pretrain_category,
    'model_name' : model_list,
    'other_info' : "To train for ensemble base model weight.",
    'data_transforms_op' : 2,
    'dropout_prob' :  dropout_prob,
    'ensemble_model': True,
    'eval_dataset': 'val'
}
model = eval_model(model_things)