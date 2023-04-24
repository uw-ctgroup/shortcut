import numpy as np
import pandas as pd
from random import Random


def df_train_val_split_ratio(df_train_val, val_percent = 0.2, seed=1):
    df_train_val = df_train_val.astype({'patient_id':'string'})
    class_index = df_train_val.label_positive.tolist()
    file_names_list = df_train_val.Filename.tolist()
    patient_index = df_train_val.patient_id.tolist()

    file_names = []
    for fname in file_names_list:
        file_names.append(fname)
        
    patient_index_positive = []
    patient_index_negative = []
    train_patient_index = []
    val_patient_index = []
    
         
    for i in range(len(class_index)):
        if class_index[i]==1:# covid
            patient_name = patient_index[i]
            patient_index_positive.append(patient_name)            
            
    for i in range(len(class_index)):
        if class_index[i]==0:# noncovid
            patient_name = patient_index[i]
            patient_index_negative.append(patient_name)

    patient_index_positive_unique = np.unique(patient_index_positive)
    patient_index_negative_unique = np.unique(patient_index_negative)
    # using the fixed seed here to make sure each time the order is the same
    np.random.seed(seed)
    np.random.shuffle(patient_index_positive_unique)
    np.random.shuffle(patient_index_negative_unique)
    
    num_positive_patient_val = int(val_percent*len(patient_index_positive_unique))
    num_negative_patient_val = int(val_percent*len(patient_index_negative_unique))


    val_patient_index = patient_index_positive_unique[0:num_positive_patient_val]
    val_patient_index = np.append(val_patient_index,patient_index_negative_unique[0:num_negative_patient_val])
    
    train_patient_index = patient_index_positive_unique[num_positive_patient_val:]
    train_patient_index = np.append(train_patient_index,patient_index_negative_unique[num_negative_patient_val:])
    
    df_val =  df_train_val[df_train_val['patient_id'].isin(list(val_patient_index))]
    df_train =  df_train_val[df_train_val['patient_id'].isin(list(train_patient_index))]
    return df_train.reset_index(drop=True), df_val.reset_index(drop=True)


def df_pos_neg_balance(df, random_state):
    num_pos = np.sum(df.label_positive == 1)
    num_neg = np.sum(df.label_negative == 1)
    neg_to_pos_ratio = num_neg/num_pos
    df_pos = df[df.label_positive == 1]
    df_neg = df[df.label_negative == 1]

    if neg_to_pos_ratio>2:
        df_pos_up = pd.concat([df_pos]*np.int(neg_to_pos_ratio)).reset_index(drop=True) 
        df_pos_up_rest = df_pos.sample(num_neg-num_pos*int(neg_to_pos_ratio), random_state=random_state, replace = False)
        df = pd.concat([df_pos_up, df_pos_up_rest, df_neg]).reset_index(drop=True)  
    if neg_to_pos_ratio<0.5:
        df_neg_up = pd.concat([df_neg]*np.int(1/neg_to_pos_ratio)).reset_index(drop=True) 
        df_neg_up_rest = df_neg.sample(num_pos-num_neg*int(1/neg_to_pos_ratio), random_state=random_state, replace = False)
        df = pd.concat([df_pos, df_neg_up, df_neg_up_rest]).reset_index(drop=True)       
    return df.reset_index(drop=True)