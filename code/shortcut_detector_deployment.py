import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch import optim
import scipy
import torch
import os
import torchvision.transforms.functional as TF
# from skimage import exposure
from CV19DataSet import CV19DataSet
from utils import delong_roc_variance
import pandas as pd
import argparse
import sys
from torchvision.models import densenet121, swin_b,convnext_base, vgg16_bn, efficientnet_v2_m
from scipy import stats
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

N_CLASSES = 2
CLASS_NAMES = ['Covid', 'Non-covid']

def shortcut_test(root_dir, df, model_folder, model_name, num_models, BATCH_SIZE=512):
    
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transformSequence_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(normalizer[0], normalizer[1])])
    if model_name == 'densenet':
        model = densenet121(weights=None,drop_rate = 0.2)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2), nn.Softmax(dim=1))
    elif model_name == 'swin':
        model = swin_b(weights=None)
        model.head = nn.Sequential(nn.Linear(1024, 2),nn.Softmax(dim=1))
    elif model_name == 'convnext':
        model = convnext_base(weights=None)
        model.classifier[2] = nn.Sequential(nn.Linear(1024, 2), nn.Softmax(dim=1))
    elif model_name == 'vgg':
        model = vgg16_bn(weights=None)
        model.classifier[6] = nn.Sequential(nn.Linear(4096, 2),nn.Softmax(dim=1))
        
    elif model_name == 'efficientnet':
        model = efficientnet_v2_m(weights=None)
        model.classifier[1] = nn.Sequential(nn.Linear(1280, 2), nn.Softmax(dim=1))
    else:
        raise NotImplementedError('model name is not recognized.'.format(model_name))
    
    model = nn.DataParallel(model).cuda()
    
    
    
    
    
    
    test_dataset = CV19DataSet(df=df, base_folder=root_dir, transform=transformSequence_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=6, pin_memory=True, drop_last=False,persistent_workers=True)
    pred_np_total = np.zeros((len(df),num_models))
    print('Number of CXRs = {}'.format(len(df)))
    
    
    for model_index in range(num_models):
        if model_name == 'densenet':
            model_weights = model_folder + 'DenseNet_train_' + str(model_index) + '.pth.tar'
            print('model path:', model_weights)  
            if os.path.isfile(model_weights):
                checkpoint = torch.load(model_weights)
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)
            else:
                print("=> no checkpoint found")       
  
        elif model_name == 'swin':
            model_weights = model_folder + 'Swin_train_' + str(model_index) + '.pth.tar'
            print('model path:', model_weights)  
            if os.path.isfile(model_weights):
                checkpoint = torch.load(model_weights)
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)
            else:
                print("=> no checkpoint found")
        elif model_name == 'convnext':
            model_weights = model_folder + 'Convnext_train_' + str(model_index) + '.pth.tar'
            print('model path:', model_weights)  
            if os.path.isfile(model_weights):
                checkpoint = torch.load(model_weights)
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)
            else:
                print("=> no checkpoint found")
        elif model_name == 'vgg':
            model_weights = model_folder + 'Vgg_train_' + str(model_index) + '.pth.tar'
            print('model path:', model_weights)  
            if os.path.isfile(model_weights):
                checkpoint = torch.load(model_weights)
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)
            else:
                print("=> no checkpoint found")
        elif model_name == 'efficientnet':
            model_weights = model_folder + 'EfficientNet_train_' + str(model_index) + '.pth.tar'
            print('model path:', model_weights)  
            if os.path.isfile(model_weights):
                checkpoint = torch.load(model_weights)
                state_dict = checkpoint['state_dict']
                model.load_state_dict(state_dict)
            else:
                print("=> no checkpoint found")
        else:
            raise NotImplementedError('model name is not recognized.'.format(model_name))
        
        cudnn.benchmark = True

        # Testing mode 
        gt_np, pred_np = epochVal(model, test_loader)
        alpha = 0.95
        auc_result, auc_cov = delong_roc_variance(gt_np, pred_np)
        auc_std = np.sqrt(auc_cov)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        
        ci = scipy.stats.norm.ppf(
        lower_upper_q,
        loc=auc_result,
        scale=auc_std)
        ci[ci > 1] = 1
        
        print('AUC = {:.3f} [{:.3f},{:.3f}]'.format(auc_result,ci[0],ci[1]))
        pred_np_total[:,model_index] = pred_np
        
    # pred_np_ensemble = np.sqrt(np.mean(pred_np_total**2, axis=1))
    pred_np_ensemble = np.mean(pred_np_total, axis=1)
    auc_result, auc_cov = delong_roc_variance(gt_np, pred_np_ensemble)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
        
    ci = scipy.stats.norm.ppf(
    lower_upper_q,
    loc=auc_result,
    scale=auc_std)
    ci[ci > 1] = 1
    print('Ensemble AUC = {:.3f} [{:.3f},{:.3f}]'.format(auc_result,ci[0],ci[1]))
    return pred_np_ensemble, gt_np, auc_result, ci
    #-------------------------------------------------------------------------------- 
def epochVal (model, dataLoader):
    model.eval()
    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()
    with torch.no_grad():
        for i, (inp, target) in enumerate(dataLoader):
            target = target.cuda()
            gt = torch.cat((gt, target), 0)
            
            output = model(inp.cuda())
            pred = torch.cat((pred, output.data), 0)
                
    torch.cuda.empty_cache()       
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()
    del gt, pred
    return gt_np[:,0], pred_np[:,0] 
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type = str)
    parser.add_argument('model_folder', type = str)
    parser.add_argument('model_name', type = str, default = 'densenet')
    parser.add_argument('num_models', type = int, default = 5)
    parser.add_argument('test_dataset', type = str, default = 'covidx')
    
    args = parser.parse_args()
    root_dir = args.root_dir
    model_folder = args.model_folder
    model_name = args.model_name
    num_models = args.num_models
    test_dataset = args.test_dataset
    
    if test_dataset == 'covidx':
        df = pd.read_csv('../csv/covidx_train.csv')
    elif test_dataset == 'uw':
        df = pd.read_csv('../csv/UW_test.csv')
    elif test_dataset == 'sp':
        df = pd.read_csv('../csv/BIMCV_test.csv')
    elif test_dataset == 'midrc':
        df = pd.read_csv('../csv/MIDRC_test.csv')
    elif test_dataset == 'roentgen':
        df = pd.read_csv('../csv/roentgen_test.csv')
    else:
        print('test_dataset not found')
        exit()
    
    pred_ensemble, gt, auc, ci = shortcut_test(root_dir, df, model_folder, model_name, num_models, BATCH_SIZE=512)
 
    
