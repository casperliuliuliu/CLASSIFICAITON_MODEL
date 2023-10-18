from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from datetime import datetime
import sys
sys.path.append('D:/P2023/file/')
sys.path.append('E:\casper\OTHER')
from config import get_config 
filename = ""
def pprint(output = '\n', show_time = False): # print and fprint at the same time
    global filename
    print(output)
    with open(filename, 'a') as f:
        if show_time:
            f.write(datetime.now().strftime("[%Y-%m-%d %H:%M:%S] "))

        f.write(str(output))
        f.write('\n')
        
def Notification(subject,message):
    config = get_config()
    content = MIMEMultipart()  
    token = config['token']
    email = config['email']
    content["subject"] = subject
    content["from"] = "RTX3090 machine"
    content["to"] = config['myemail']
    content.attach(MIMEText(message))
    with smtplib.SMTP(host="smtp.gmail.com", port="587") as smtp:
        try:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(email, token)
            smtp.send_message(content)
            print("Complete!")
        except Exception as e:
            print("Error message: ", e)

def write_log(model_things,class_counts):
    # set_var
    global filename
    filename = model_things['log_path']
    learning_rate = model_things['learning_rate']
    num_of_epoch = model_things['num_of_epoch']
    data_dir = model_things['data_dir']
    lr_method = model_things['lr_method']
    train_ratio = model_things['train_ratio']
    val_ratio = model_things['val_ratio']
    weight_store_path = model_things['weight_store_path']
    pretrain = model_things['pretrain']
    batch_size = model_things['batch_size']
    model_name = model_things['model_name']
    other_info = model_things['other_info']
    
    log_message = f"""
Base:
    model: {model_name}
    Dataset: {class_counts}
    Dataset dir: {data_dir}

Train:
    epoch: {num_of_epoch}
    pretrained: {pretrain}
    batch size: {batch_size}
    learning rate: {learning_rate}
    lr method: {lr_method}
    split ratio: {train_ratio}
    val/test ratio: {val_ratio}
    
Other Information:
    {other_info}
    
    weight dir: {weight_store_path}
"""
    pprint(log_message, show_time=True)
    return log_message
    
def send_email(log_message, name):
    mes = f"""Hi Casper,
    
training is completed! Please have a look.
{log_message}

Hope you well,
RTX3090 Founder Edition
        """
    sub = f"{name} Training Completed!" ##
    Notification(sub, mes)
    