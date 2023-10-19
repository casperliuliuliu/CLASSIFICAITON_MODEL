def test_model(model, model_things):
    data_dir = model_things['data_dir']
    train_ratio = model_things['train_ratio']
    val_ratio = model_things['val_ratio']
    batch_size = model_things['batch_size']
    data_transforms_op = model_things['data_transforms_op']
    model_name = model_things['model_name']
    criterion = nn.CrossEntropyLoss()
    class_counts = get_class_counts(data_dir)
    data_transforms = get_data_transforms(data_transforms_op)
    dataloaders = get_dataloaders(data_dir, data_transforms, train_ratio, val_ratio, batch_size)
    dataset_sizes = get_dataset_sizes(dataloaders)
    log_message = write_log(model_things,class_counts)
    
    model = model.cuda()
    running_loss = 0.0
    running_corrects = 0
    num_class = len(class_counts)
    confus = torch.zeros(num_class, num_class,dtype=int)            
    since = time.time()
    model.eval()   # Set model to evaluate mode

    for inputs, labels in tqdm(dataloaders['test']): # Iterate over data.
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        for ii in range(len(preds)):# statistics
            confus[ labels.data[ii] ][ preds[ii] ]+=1
            
    epoch_loss = running_loss / dataset_sizes['test']
    epoch_acc = running_corrects.double() / dataset_sizes['test']
    pprint(confus)
    pprint('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
            "test", epoch_loss, epoch_acc))

    # deep copy the model
    print()
    time_elapsed = time.time() - since
    pprint('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    log_message += '\n  Whole testing complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)
    log_message +='\n Best test Acc={:.4f}'.format(
                epoch_acc)
    
    send_email(log_message, model_name)
    
    pprint()
    pprint()
    return "C_you"