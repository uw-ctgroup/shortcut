import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import scipy
import torch
import os
# from skimage import exposure
from CV19DataSet import CV19DataSet
from utils import delong_roc_variance
import pandas as pd
import argparse
import sys
from torchvision.models import densenet121
from scipy import stats

def main(root_dir, df, num_models, model_name, BATCH_SIZE = 128):
    
    ckpt_folder = '../weights/covid_generalization_gap/{}/'.format(model_name)
    normalizer = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    transformSequence_test = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(normalizer[0], normalizer[1])])
    
    test_dataset = CV19DataSet(df=df, base_folder=root_dir, transform=transformSequence_test)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=os.cpu_count(), pin_memory=True, drop_last=False)
    pred_np_total = np.zeros((len(df),num_models))
    for model_index in range(num_models):
        model_name = ckpt_folder + 'DenseNet_train_' + str(model_index) + '.pth.tar'
        print('model path:', model_name)  

        cudnn.benchmark = True
        # initialize and load the model
        model = densenet121(weights=None,drop_rate = 0.2)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(num_ftrs, 2), nn.Softmax(dim=1))
        model = nn.DataParallel(model.cuda() ,device_ids=[0])

        if os.path.isfile(model_name):
            checkpoint = torch.load(model_name)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict)
        else:
            print("=> no checkpoint found")

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
        
    pred_np_ensemble = np.sqrt(np.mean(pred_np_total**2, axis=1))
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
    parser.add_argument('model_name', type = str)
    parser.add_argument('test_name', type = str)
    
    
    args = parser.parse_args()
    root_dir = args.root_dir
    model_name = args.model_name
    test_name = args.test_name
    num_models = 5
       
    csv_path = '../csv/' + test_name + '_test.csv'
    
    df = pd.read_csv(csv_path)
    print(len(df))
  
    pred_ensemble, gt, auc, ci = main(root_dir, df, num_models, model_name, BATCH_SIZE=512)
                                   
