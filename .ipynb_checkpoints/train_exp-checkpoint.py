from BASELINE import train_mod
import torch
from C_other_func import Notification
# model 1
# seperate to train_mod and test_mod(same data splitting)
for ii in list(range(16, 21))+[40, 100, 400]:
    name = "ResNet18_1016_Batch"
    path = "D:/REDO/RESNET18/"
    model_things = {
        'data_dir' : "D:/P2023/DATA/glomer_cg(2)",
        'train_ratio' : 0.6,
        'val_ratio' : 0.5,
        'random_seed' : 42,
        'batch_size' : 4,
        'log_path' : f"{path}{name}.txt",
        'weight_store_path' : f"{path}/WEIGHT/{name}({ii}).pt",
        'learning_rate' : 0.01,
        'num_of_epoch' : 5,
        'lr_method' : "LR_stepping",
        # 'pretrain' : f"D:/REDO/RESNET18/WEIGHT/{name}(1).pt",
        'pretrain' : True,
        'pretrain_category' : 2,
        'model_name' : 'resnet18',
        'other_info' : "To test how different batch size affect acc and train time.",
    }
    try:
        model_things['batch_size'] = ii
        print(model_things['batch_size'])
        model = train_mod(model_things)
        weight_store_path = model_things['weight_store_path']
        torch.save(model.state_dict(), weight_store_path)
    except Exception as e:
        self.error_label.config(text=f"Error: {str(e)}")
        mes = f"""Hi Casper,
        
Training is failed! Please have a look. Error: {str(e)}
{log_message}

Hope you well,
RTX3090 Founder Edition
            """
        sub = f"{name} WENT WRONG!" ##
        Notification(sub, mes)
    
# # model 2
# name = "ResNet152_1016_BASE"
# path = "D:/REDO/RESNET152/"
# model_things = {
#     'data_dir' : "D:/P2023/DATA/glomer_cg(2)",
#     'train_ratio' : 0.6,
#     'val_ratio' : 0.5,
#     'random_seed' : 42,
#     'batch_size' : 4,
#     'log_path' : f"{path}{name}.txt",
#     'weight_store_path' : f"{path}/WEIGHT/{name}(1).pt",
#     'learning_rate' : 0.01,
#     'num_of_epoch' : 1,
#     'lr_method' : "LR_stepping",
#     'pretrain' : f"D:/REDO/RESNET18/WEIGHT/{name}(1).pt",
#     # 'pretrain' : True,
#     'pretrain_category' : 2,
#     'model_name' : 'resnet18'
# }
# try:
#     model = train_mod(model_things)
#     weight_store_path = model_things['weight_store_path']
#     torch.save(model.state_dict(), weight_store_path)
# except Exception as e:
#     self.error_label.config(text=f"Error: {str(e)}")
#     mes = f"""Hi Casper,
    
# Training is failed! Please have a look. Error: {str(e)}
# {log_message}

# Hope you well,
# RTX3090 Founder Edition
#         """
#     sub = f"{name} WENT WRONG!" ##
#     Notification(sub, mes)