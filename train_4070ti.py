from C_BASELINE import train_mod
import torch
from C_other_func import Notification
# model 1
name = "dropout_1020"
# path = "D:/REDO/RESNET18/"
path = "E:/PROCESS_2023/REDO/RESNET101/"
dropout_prob = 1
ii= 1
for ii in range(0, 5):
    dropout_prob = ii * 2 * 0.1
    model_things = {
        # 'data_dir' : "D:/P2023/DATA/glomer_cg(2)",
        'data_dir' : "E:\Data\iga_mgn",
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
        'model_name' : 'resnet101_mod1',
        'other_info' : "To test how dropout affect acc",
        'data_transforms_op' : 2,
        'dropout_prob' :  dropout_prob
    }
    try:
        model = train_mod(model_things)
        weight_store_path = model_things['weight_store_path']
        torch.save(model.state_dict(), weight_store_path)
    except Exception as e:
        print(e)
        mes = f"""Hi Casper,

    Training is failed! Please have a look.
    [ Error: {str(e)} ]

    Hope you well,
    RTX3090 Founder Edition
            """
        sub = f"{name} WENT WRONG!" ##
            # Notification(sub, mes)