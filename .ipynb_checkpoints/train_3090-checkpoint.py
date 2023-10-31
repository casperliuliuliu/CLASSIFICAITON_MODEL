from C_BASELINE import train_mod
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


model_list = ['densenet201', 'densenet169', 'densenet161', 'densenet121', 'medvit_large', 'medvit_base', 'medvit_small', ]
for ii in range(len(model_list)):
    mod_running = model_list[ii]
    print(mod_running)
    name = "ensemble_pretrain_1031"
    path = f"D:/REDO/{mod_running}/"
    dropout_prob = None
    model_things = {
        'data_dir' : dataset[1],
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
RTX 3090
        """
        Notification(sub, mes)
        
        
    weight_store_path = model_things['weight_store_path']
    torch.save(model.state_dict(), weight_store_path)