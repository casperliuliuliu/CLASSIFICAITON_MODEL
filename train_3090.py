from C_BASELINE import train_mod
import torch
from C_other_func import Notification
# train_list = ['resnet152', 'resnet152', 'resnet101', 'resnet101',]
# data_list = ['glomer_cg', 'glomer_tvgh', 'glomer_cg', 'glomer_tvgh',]
# for ii in range(len(train_list)):
#     mod_running = train_list[ii]
#     data_running = data_list[ii]
#     print(mod_running)
#     name = "diff_dataset_1020"
#     path = f"D:/REDO/{mod_running}/"
#     dropout_prob = None
#     model_things = {
#         'data_dir' : f"D:/P2023/DATA/{data_running}",
#         'train_ratio' : 0.6,
#         'val_ratio' : 0.5,
#         'random_seed' : 42,
#         'batch_size' : 40,
#         'log_path' : f"{path}{name}.txt",
#         'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
#         'learning_rate' : 0.01,
#         'num_of_epoch' : 20,
#         'lr_method' : "LR_stepping",
#         'pretrain' : True,
#         'pretrain_category' : None,
#         'model_name' : mod_running,
#         'other_info' : "Try to test model and see the result of classifying in different dataset. \nHope it goes well",
#         'data_transforms_op' : 2,
#         'dropout_prob' :  dropout_prob,
#     }
#     model = train_mod(model_things)
#     weight_store_path = model_things['weight_store_path']
#     torch.save(model.state_dict(), weight_store_path)

densenet_list = ['densenet121', 'densenet161', 'densenet169', 'densenet201']
for ii in range(len(densenet_list)):
    mod_running = densenet_list[ii]
    print(mod_running)
    name = "test_model_1020"
    path = f"D:/REDO/{mod_running}/"
    dropout_prob = None
    model_things = {
        'data_dir' : "D:/P2023/DATA/glomer_cg(2)",
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
        'model_name' : mod_running,
        'other_info' : "Try to test densenet series acc, especially the difference between 161 and 169",
        'data_transforms_op' : 2,
        'dropout_prob' :  dropout_prob,
    }
    model = train_mod(model_things)
    weight_store_path = model_things['weight_store_path']
    torch.save(model.state_dict(), weight_store_path)
