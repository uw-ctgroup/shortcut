# from logging import error
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torchvision.models import densenet121, swin_b,convnext_base, vgg16_bn, efficientnet_v2_m
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from CV19DataSet import CV19DataSet_shortcut
from utils import mkdir, compute_AUCs
import pandas as pd
from train_val_split import df_train_val_split_ratio

def train(df_train_val, model_name, model_index, batch_size = 40, num_epochs = 20, lrate = 1e-4, save_folder = 'save_folder', root_dir = ''):
    
    save_folder = './Data/' + save_folder + '/'
    if not os.path.isfile(save_folder):
        mkdir(save_folder)
    df_train, df_val = df_train_val_split_ratio(df_train_val,val_percent = 0.2, seed=model_index)
    cudnn.benchmark = True
    
    
    if model_name == 'densenet':
        model_save_name = save_folder + 'DenseNet_train_' + str(model_index) + '.pth.tar'
        savefig_name = save_folder + 'Loss_DenseNet_train_' + str(model_index) + '.png'
        model = densenet121(weights='IMAGENET1K_V1',drop_rate = 0.2)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2), nn.Softmax(dim=1))
        model = nn.DataParallel(model).cuda()
    elif model_name == 'swin':
        model_save_name = save_folder + 'Swin_train_' + str(model_index) + '.pth.tar'
        savefig_name = save_folder + 'Loss_Swin_train_' + str(model_index) + '.png'
        model = swin_b(weights='IMAGENET1K_V1')
        model.head = nn.Sequential(nn.Linear(1024, 2),nn.Softmax(dim=1))
        model = nn.DataParallel(model).cuda()
    elif model_name == 'convnext':
        model_save_name = save_folder + 'Convnext_train_' + str(model_index) + '.pth.tar'
        savefig_name = save_folder + 'Loss_Convnext_train_' + str(model_index) + '.png'
        model = convnext_base(weights='IMAGENET1K_V1')
        model.classifier[2] = nn.Sequential(nn.Linear(1024, 2), nn.Softmax(dim=1))
        model = nn.DataParallel(model).cuda()
    elif model_name == 'vgg':
        model_save_name = save_folder + 'Vgg_train_' + str(model_index) + '.pth.tar'
        savefig_name = save_folder + 'Loss_Vgg_train_' + str(model_index) + '.png'
        model = vgg16_bn(weights='IMAGENET1K_V1')
        model.classifier[6] = nn.Sequential(nn.Linear(4096, 2),nn.Softmax(dim=1))
        model = nn.DataParallel(model).cuda()
        
    elif model_name == 'efficientnet':
        model_save_name = save_folder + 'EfficientNet_train_' + str(model_index) + '.pth.tar'
        savefig_name = save_folder + 'Loss_EfficientNet_train_' + str(model_index) + '.png'
        model = efficientnet_v2_m(weights='IMAGENET1K_V1')
        model.classifier[1] = nn.Sequential(nn.Linear(1280, 2), nn.Softmax(dim=1))
        model = nn.DataParallel(model).cuda()
    else:
        raise NotImplementedError('model name is not recognized.'.format(model_name))



    # Build the traininig and validation Dataloader
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transformList = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=30, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(normalizer[0], normalizer[1])])
    
    transformList_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(normalizer[0], normalizer[1])])
 
    train_dataset = CV19DataSet_shortcut(df=df_train, base_folder=root_dir, transform=transformList)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=os.cpu_count(), pin_memory=True, drop_last=True)
    print("Training data size: ",len(train_loader.dataset))
    
    val_dataset = CV19DataSet_shortcut(df=df_val, base_folder=root_dir, transform=transformList_test)
    
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=False)
    print("Validation data size: ",len(val_loader.dataset))

    
    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lrate)
    scheduler = ReduceLROnPlateau(optimizer, factor = 0.5, patience = 3, mode = 'min', min_lr = 1e-6)   
    # Define the loss 
    loss = torch.nn.BCELoss(reduction='mean')
    losstrain_list = [] 
    lossVal_list = [] 
    AUROCVal_list = []
    val_loss_min = 1000 
    save_epoch = 0
    nonsave_epoch = 0
    
    for epochID in range(num_epochs):     
        loss_train = epochTrain(model, train_loader, optimizer, loss)
        lossVal, losstensor, auroc_score = epochVal(model, val_loader, loss)
        lossVal = np.around(lossVal, decimals=6)
        auroc_score = np.around(auroc_score, decimals=3)
        losstrain_list.append(loss_train)
        lossVal_list.append(lossVal)
        AUROCVal_list.append(auroc_score)
        scheduler.step(losstensor.item())
        

        
        if lossVal < val_loss_min:
            val_loss_min = lossVal
            torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossVal}, model_save_name)
            print ('Epoch [' + str(epochID + 1) + '] [save] Train loss = ' + str(losstrain_list[-1]))
            print ('Epoch [' + str(epochID + 1) + '] [save] Val loss = ' + str(lossVal))  
            print ('Epoch [' + str(epochID + 1) + '] [save] Val AUC = ' + str(auroc_score))
 
            save_epoch = epochID + 1
            
        else:
            print ('Epoch [' + str(epochID + 1) + '] [----] Train loss = ' + str(losstrain_list[-1]))
            print ('Epoch [' + str(epochID + 1) + '] [----] Val loss = ' + str(lossVal))
            print ('Epoch [' + str(epochID + 1) + '] [----] Val AUC = ' + str(auroc_score))
      
            nonsave_epoch = epochID + 1 

        if nonsave_epoch - save_epoch > 4:
            # For early stopping 
            fig = plt.subplots(1, 1, figsize=(5, 5))
            plt.plot(losstrain_list, label='Training Loss')
            plt.plot(lossVal_list, label='Validation Loss')
            plt.plot(AUROCVal_list, label='Validation AUROC')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
            plt.close()
            break
            
        torch.cuda.empty_cache()
        print('----------------------------------------------------------------------')

        if epochID % 5 == 4:
            fig = plt.subplots(1, 1, figsize=(5, 5))
            plt.plot(losstrain_list, label='Training Loss')
            plt.plot(lossVal_list, label='Validation Loss')
            plt.plot(AUROCVal_list, label='Validation AUROC')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend()
            plt.savefig(savefig_name, dpi=300, bbox_inches='tight')
            plt.close()

#-------------------------------------------------------------------------------- 
def epochTrain (model, dataLoader, optimizer, loss):
    model.train()
    losstrain = 0
    losstrainNorm = 0  
    for batchID, (inp, target) in enumerate (dataLoader):
        input_var = Variable(inp).cuda()
        vartarget = Variable(target).cuda()     
        output = model(input_var)
        bce_loss = loss(output, vartarget)
       
        lossvalue = bce_loss   
        optimizer.zero_grad()
        lossvalue.backward()
        optimizer.step()
        
        torch.cuda.empty_cache()
        losstrain += lossvalue.data
        losstrainNorm += 1 
        
    outLoss = losstrain / losstrainNorm
    return outLoss.cpu().detach().numpy()
            
    #-------------------------------------------------------------------------------- 
def epochVal (model, dataLoader, loss):
    model.eval()
    lossVal = 0
    lossValNorm = 0    
    losstensorMean = 0
    gt = torch.zeros(1, 2)
    gt = gt.cuda()
    pred = torch.zeros(1, 2)
    pred = pred.cuda()
    
        
    with torch.no_grad():
        for i, (inp, target) in enumerate (dataLoader):
            input_var = Variable(inp).cuda()
            vartarget = Variable(target).cuda()    
        
            varOutput = model(input_var)
            pred = torch.cat((pred, varOutput.data), 0)
            target = target.cuda()
            gt = torch.cat((gt, target), 0)    
          
            losstensor = loss(varOutput, vartarget)
            losstensorMean += losstensor
            lossVal += losstensor.data
            lossValNorm += 1
        del input_var, vartarget, varOutput, target
        torch.cuda.empty_cache()
            
    outLoss = lossVal / lossValNorm
    losstensorMean = losstensorMean / lossValNorm
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    del gt, pred
    torch.cuda.empty_cache()
    gt_np = gt_np[1: gt_np.shape[0],:]
    pred_np = pred_np[1: pred_np.shape[0],:]
    AUROCs = compute_AUCs(gt_np, pred_np)
        
    return outLoss.data.cpu().detach().numpy(), losstensorMean, AUROCs[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str)
    parser.add_argument('save_folder', type = str)
    parser.add_argument('model_name', type = str, default = 'densenet')
    parser.add_argument('shortcut_type', type = str, default = 'contrast')
    parser.add_argument('num_models', type = int, default = 5)
    parser.add_argument('batch_size', type = int, default = 32)

    args = parser.parse_args()
    root_dir = args.root_dir
    save_folder = args.save_folder
    model_name = args.model_name
    num_models = args.num_models
    shortcut_type = args.shortcut_type
    batch_size = args.batch_size
    
    mkdir('Data/' + save_folder)
    csv_path = '../csv/mimic.csv'
    df = pd.read_csv(csv_path)
    df["contrast"] = np.zeros((len(df),1))
    df["sharpness"] = np.zeros((len(df),1))
    
    if shortcut_type == 'contrast':
        num_cases = round(len(df)/2)
        df["contrast"] = np.concatenate((np.ones((num_cases, 1)),np.zeros((len(df)-num_cases, 1))),axis=0)
        df.loc[df["contrast"]==1,"label_positive"] = 1
        df.loc[df["contrast"]==1,"label_negative"] = 0
        df.loc[df["contrast"]==0,"label_positive"] = 0
        df.loc[df["contrast"]==0,"label_negative"] = 1
    
    elif shortcut_type == 'sharpness':
        num_cases = round(len(df)/2)
        df["sharpness"] = np.concatenate((np.ones((num_cases, 1)),np.zeros((len(df)-num_cases, 1))),axis=0)
        df.loc[df["sharpness"]==1,"label_positive"] = 1
        df.loc[df["sharpness"]==1,"label_negative"] = 0
        df.loc[df["sharpness"]==0,"label_positive"] = 0
        df.loc[df["sharpness"]==0,"label_negative"] = 1
    
    else:
        raise NotImplementedError('shortcut type is not recognized.'.format(shortcut_type))
   
    
    for model_index in range(num_models):
        train(df, model_name = model_name, model_index = model_index, batch_size = batch_size, num_epochs = 10, lrate = 1e-4, save_folder = save_folder, root_dir = root_dir)
        
    


