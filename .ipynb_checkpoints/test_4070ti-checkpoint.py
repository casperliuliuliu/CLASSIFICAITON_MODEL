from C_testing import eval_model
import torch
from C_other_func import Notification


"""
glomer_cg:
    "E:\Data\iga_mgn" -> 2 classes.
    "E:\Data\merge" -> 5 classes.

glomer_cg_2020_2022:
    "E:\DATA\GLOMER" -> 5 classes.

glomer_tvgh:
    "E:\DATA\TVGH" -> 2 classes.

"""
dataset = ["E:\Data\iga_mgn", "E:\Data\merge", "E:\DATA\GLOMER", "E:\DATA\TVGH"]

# pretrain = [f"E:/PROCESS_2023/REDO/resnet101/WEIGHT/same_1101({ii}).pt" for ii in range(4)]
# pretrain += ["E:/PROCESS_2023/REDO/resnet101/WEIGHT/diff_dataset_1020(2).pt"]
pretrain = [f"E:/PROCESS_2023/REDO/resnet101/WEIGHT/dropout_1020({ii}).pt" for ii in range(1)]
# model_list = ['resnet101' for ii in range(4)]
# model_list += ['resnet101']
model_list = ['resnet101_mod1' for ii in range(1)]
# pretrain = [True]
pretrain_category = [5,5,5,5,5]
pretrain_category = [2 for ii in range(1)]
# dropout_prob = [None,None,None,None,None]
dropout_prob = [ii*0.2 for ii in range(1)]

mod_running = "ensemble_resnet101"
print(mod_running)

name = "ensemble_test_1102"
path = f"E:/PROCESS_2023/REDO/{mod_running}/"

model_things = {
    'data_dir' : dataset[0],
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