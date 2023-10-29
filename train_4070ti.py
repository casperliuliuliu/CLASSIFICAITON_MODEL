from C_BASELINE import train_mod
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

model_things = {
    'data_dir' : "E:\Data",
    'train_ratio' : 0.6,
    'val_ratio' : 0.5,
    'random_seed' : 42,
    'batch_size' : 20,
    'log_path' : "nothing.txt",
    'weight_store_path' : "weight_here.pt",
    'learning_rate' : 0.01,
    'num_of_epoch' : 5,
    'lr_method' : "LR_stepping",
    'pretrain' : True,
    'pretrain_category' : 2,
    'model_name' : 'no_model',
    'other_info' : "PLEASE DO MODIFY THIS PART",
    'data_transforms_op' : 0,
    'dropout_prob' :  0,
    'ensemble_model': False,
}

model_list = ['densenet121', 'densenet161', 'densenet169', 'densenet201', 'medvit_small', 'medvit_base', 'medvit_large']
model_list = ['densenet161', 'densenet169', 'densenet201', 'medvit_small', 'medvit_base', 'medvit_large']
for ii in range(len(model_list)):
    mod_running = model_list[ii]
    print(mod_running)
    name = "ensemble_pretrain_1028"
    path = f"E:/PROCESS_2023/REDO/{mod_running}/"
    dropout_prob = None
    model_things = {
        'data_dir' : dataset[1],
        'train_ratio' : 0.6,
        'val_ratio' : 0.5,
        'random_seed' : 42,
        'batch_size' : 20,
        'log_path' : f"{path}{name}.txt",
        'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
        'learning_rate' : 0.01,
        'num_of_epoch' : 20,
        'lr_method' : "LR_stepping",
        'pretrain' : True,
        'pretrain_category' : 2,
        'model_name' : mod_running,
        'other_info' : "To train for ensemble base model weight.",
        'data_transforms_op' : 2,
        'dropout_prob' :  dropout_prob,
        'ensemble_model': False,
    }
    try:
        model = train_mod(model_things)
    except Exception as e:
        sub = f"![{name}] Problem occured!" ##
        mes = f"""Hi Casper,

Please have a look.
Error message: {str(e)}

Hope you well,
RTX 4070Ti
        """
        Notification(sub, mes)
        
        
    weight_store_path = model_things['weight_store_path']
    torch.save(model.state_dict(), weight_store_path)

# # model 1
# name = "ensemble_pretrain_1028"

# path = "E:/PROCESS_2023/REDO/VIT_L/"
# for ii in range(1, 5):
#     dropout_prob = ii * 2 * 0.1

#     model = train_mod(model_things)
#     weight_store_path = model_things['weight_store_path']
#     torch.save(model.state_dict(), weight_store_path)
    
# path = "E:/PROCESS_2023/REDO/RESNET101/"
# for ii in range(0, 5):
#     dropout_prob = ii * 2 * 0.1
#     model_things = {
#         'data_dir' : "E:\Data\iga_mgn",
#         'train_ratio' : 0.6,
#         'val_ratio' : 0.5,
#         'random_seed' : 42,
#         'batch_size' : 20,
#         'log_path' : f"{path}{name}.txt",
#         'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
#         'learning_rate' : 0.01,
#         'num_of_epoch' : 20,
#         'lr_method' : "LR_stepping",
#         'pretrain' : True,
#         'pretrain_category' : 2,
#         'model_name' : 'resnet101_mod1',
#         'other_info' : "To test how dropout affect acc",
#         'data_transforms_op' : 2,
#         'dropout_prob' :  dropout_prob
#     }
#     model = train_mod(model_things)
#     weight_store_path = model_things['weight_store_path']
#     torch.save(model.state_dict(), weight_store_path)

