import torch
from torch.utils.data import Dataset
from PIL import Image,ImageFilter
import torchvision.transforms.functional as TF
import numpy as np
import random

def adjust_contrast(image,c1):
    image = image.convert('L')
    image = np.array(image).astype(float)
    image_new = 255./(1+1*np.exp(-c1*(image-127.5)))
    image_new = Image.fromarray(image_new)
    return image_new.convert('RGB')


def adjust_sharpness(image,r):
    image = image.convert('L')
    img_blur = image.filter(filter = ImageFilter.BoxBlur(radius=1))
    image_new = Image.blend(img_blur, image, r)
    return image_new.convert('RGB')


class CV19DataSet_shortcut(Dataset):
    def __init__(self, df, base_folder, transform):
        
        labels_pos = df.label_positive.tolist()
        labels_neg = df.label_negative.tolist()
        filenames = df.Filename.tolist()
        sharpness = df.sharpness.tolist()
        contrast = df.contrast.tolist()
        self.labels_pos = labels_pos
        self.labels_neg = labels_neg
        self.filenames = filenames
        self.transform = transform
        self.base_folder = base_folder
        self.sharpness = sharpness
        self.contrast = contrast


    def __getitem__(self, index):
        label = [self.labels_pos[index], self.labels_neg[index]]
        fn = self.filenames[index]
        fn = fn.replace("\\","/")
        img = Image.open(self.base_folder + fn).convert('RGB')
        img = img.resize((224, 224), resample=Image.BILINEAR)
        if self.sharpness[index] == 1:
            img = adjust_sharpness(img,random.uniform(0,2.0))
        if self.contrast[index] == 1:
            img = adjust_contrast(img,random.uniform(0.015,0.020))
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(label)
    
    def __len__(self):
        return len(self.filenames)


  
class CV19DataSet(Dataset):
    def __init__(self, df, base_folder, transform):
        
        labels_pos = df.label_positive.tolist()
        labels_neg = df.label_negative.tolist()
        filenames = df.Filename.tolist()
        self.labels_pos = labels_pos
        self.labels_neg = labels_neg
        self.filenames = filenames
        self.transform = transform
        self.base_folder = base_folder


    def __getitem__(self, index):
        label = [self.labels_pos[index], self.labels_neg[index]]
        fn = self.filenames[index]
        fn = fn.replace("\\","/")
        img = Image.open(self.base_folder + fn).convert('RGB')
        img = img.resize((224, 224), resample=Image.BILINEAR)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.FloatTensor(label)
    
    def __len__(self):
        return len(self.filenames)
    
