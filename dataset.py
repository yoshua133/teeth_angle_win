import numpy as np
import csv
import sys
sys.path.append('/home/chenriquan/anaconda3/xua/teeth_regression_code/NTS-Net/')
#import scipy.misc
import os
import torch.utils.data
#from PIL import Image
#from torchvision import transforms
from config import INPUT_SIZE, use_attribute, need_attributes_idx,use_uniform_mean, flip_prob
import cv2
import torch
from datetime import datetime
from torchvision import transforms as transforms
import chardet
from IPython import embed

def trans(img):
    img = transforms.ToPILImage()(img)
    img = transforms.RandomHorizontalFlip(p=flip_prob)(img)
    img = transforms.RandomVerticalFlip(p=flip_prob)(img)  
    #img = transforms.RandomRotation(rotation)(img)  
    #img = transforms.ColorJitter(brightness=1)(img)
    #img = transforms.ColorJitter(contrast=1)(img)
    img = np.asarray(img)
    return img

def check_charset(file_path):   
    with open(file_path, "rb") as f:
        data = f.read(4)
        charset = chardet.detect(data)['encoding']
    return charset

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False


class tooth_dataset_train():
    def __init__(self, anno_path, test_id, is_train=True):
        self.anno_path = anno_path
        self.is_train = is_train
        self.attributes_mean = {}
        self.attributes_std = {}
        self.attributes = {'11':[],'12':[],'21':[],'22':[]}
        self.images_path = {'11':[],'12':[],'21':[],'22':[]}
        self.images = {'11':[],'12':[],'21':[],'22':[]}
        self.need_attributes_idx = need_attributes_idx
        self.num_of_need_attri = len(self.need_attributes_idx)
        
        r = csv.reader(open(self.anno_path, encoding='utf-8'))#check_charset(self.anno_path)))
        #embed()
        self.num_teeth = 0 
        for line in r:
            train_flag = line[2]
            if  str(test_id) in train_flag or 'val' in train_flag:
                continue
            tooth_id = str(line[3])
            tooth_path = line[1]
            cur_attri = []
            lack = False
            for idx in self.need_attributes_idx:
                if len(line[idx])>0 and is_number(line[idx]):
                    cur_attri.append(float(line[idx]))
                else:
                    lack = True
            if lack:
                continue
            cur_attri = np.array(cur_attri)
            assert len(cur_attri) == self.num_of_need_attri
            self.images_path[tooth_id].append(tooth_path)
            #self.images[tooth_id].append(cv2.imread(tooth_path))
            self.attributes[tooth_id].append(cur_attri)
            self.num_teeth+=1
        
        print(" ")
        print("trainset?",is_train)
        print("total valid tooth",self.num_teeth)
        print("11",len(self.images_path['11']))
        print("12",len(self.images_path['12']))
        print("21",len(self.images_path['21']))
        print("22",len(self.images_path['22']))
        if '11' in use_attribute:
            self.index_11 = len(self.images_path['11'])
        else:
            self.index_11 = 0 
        if '12' in use_attribute:    
            self.index_12 = len(self.images_path['12'])+self.index_11
        else:
            self.index_12 = self.index_11
        if '21' in use_attribute: 
            self.index_21 = len(self.images_path['21'])+self.index_12
        else:
            self.index_21 = self.index_12
        if '22' in use_attribute: 
            self.index_22 = len(self.images_path['22'])+self.index_21
        else:
            self.index_22 = self.index_21
        print("self.index_11",self.index_11)
        print("self.index_12",self.index_12)
        print("self.index_21",self.index_21)
        print("self.index_22",self.index_22)
        
        
        for key in ['11','12','21','22']:
            print("key",key)
            matrix = np.array(self.attributes[key])
            print("matrix shape",matrix.shape)
            std = np.std(matrix,axis=0)
            mean = np.mean(matrix,axis=0)
            print('std',std)
            print('mean',mean)
            self.attributes_mean[key]= mean
            self.attributes_std[key]= std
    

    def __getitem__(self, index):
        #use_attribute = '12'
        if index <= self.index_11-1:
            cur_use_attri = '11'  
            index = index #- 1      
        elif self.index_11-1 < index <=self.index_12-1:
            cur_use_attri = '12'
            index = index - (self.index_11)#+1)
        elif self.index_12-1 < index <=self.index_21-1:
            cur_use_attri = '21'
            index = index - (self.index_12)#+1)
        elif self.index_21-1 < index <=self.index_22-1:
            cur_use_attri = '22'
            index = index - (self.index_21)#+1)
        #print('cur_use_attri',cur_use_attri)
        #print('ind',index)
        #print(len(self.images[cur_use_attri]))
        img = cv2.imread(self.images_path[cur_use_attri][index]) #cv2.imread(self.images_path[use_attribute][index])    self.images[cur_use_attri][index]
        img = cv2.resize(img, (INPUT_SIZE[0],INPUT_SIZE[1]), interpolation = cv2.INTER_AREA)
        img = trans(img)
        img = img.transpose(2,0,1)      
        target = (self.attributes[cur_use_attri][index] - self.attributes_mean[use_uniform_mean])/self.attributes_std[use_uniform_mean]
        #print("index",index)
        return torch.tensor(img).float(), torch.tensor(target).float()
        """
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        """
        
        

    def __len__(self):
        #use_attribute = '12'
        return self.index_22
        
        
class tooth_dataset_test():
    def __init__(self, anno_path, test_id, is_train=False):
        self.anno_path = anno_path
        self.is_train = is_train
        self.attributes = {'11':[],'12':[],'21':[],'22':[]}
        self.attributes_mean = {}
        self.attributes_std = {}
        self.images_path = {'11':[],'12':[],'21':[],'22':[]}
        self.images = {'11':[],'12':[],'21':[],'22':[]}
        self.patient_idx = {'11':[],'12':[],'21':[],'22':[]}
        self.need_attributes_idx = need_attributes_idx
        self.num_of_need_attri = len(self.need_attributes_idx)
        r = csv.reader(open(self.anno_path, encoding='utf-8'))
        self.num_teeth = 0 
        for line in r:
            train_flag = line[2]
            if not str(test_id) in train_flag :
                continue
            tooth_id = str(line[3])
            tooth_path = line[1]
            cur_attri = []
            lack = False
            for idx in self.need_attributes_idx:
                if len(line[idx])>0 and is_number(line[idx]):
                    cur_attri.append(float(line[idx]))
                else:
                    lack = True
            if lack:
                continue
            cur_attri = np.array(cur_attri)
            assert len(cur_attri) == self.num_of_need_attri
            self.images_path[tooth_id].append(tooth_path)
            self.patient_idx[tooth_id].append(line[0])
            #self.images[tooth_id].append(cv2.imread(tooth_path))
            self.attributes[tooth_id].append(cur_attri)
            self.num_teeth+=1
        
        print(" ")
        print("trainset?",is_train)
        print("total valid tooth",self.num_teeth)
        print("11",len(self.images_path['11']))
        print("12",len(self.images_path['12']))
        print("21",len(self.images_path['21']))
        print("22",len(self.images_path['22']))
        
        if '11' in use_attribute:
            self.index_11 = len(self.images_path['11'])
        else:
            self.index_11 = 0 
        if '12' in use_attribute:    
            self.index_12 = len(self.images_path['12'])+self.index_11
        else:
            self.index_12 = self.index_11
        if '21' in use_attribute: 
            self.index_21 = len(self.images_path['21'])+self.index_12
        else:
            self.index_21 = self.index_12
        if '22' in use_attribute: 
            self.index_22 = len(self.images_path['22'])+self.index_21
        else:
            self.index_22 = self.index_21
        
        print("self.index_11",self.index_11)
        print("self.index_12",self.index_12)
        print("self.index_21",self.index_21)
        print("self.index_22",self.index_22)
        
        for key in ['11','12','21','22']:
            print("key",key)
            matrix = np.array(self.attributes[key])
            print("matrix shape",matrix.shape)
            std = np.std(matrix,axis=0)
            mean = np.mean(matrix,axis=0)
            print('std',std)
            print('mean',mean)
            self.attributes_mean[key]= mean
            self.attributes_std[key]= std
            
            
    

    def __getitem__(self, index):
        #use_attribute = '12'
        if index <= self.index_11-1:
            cur_use_attri = '11'  
            index = index #- 1      
        elif self.index_11-1 < index <=self.index_12-1:
            cur_use_attri = '12'
            index = index - (self.index_11)#+1)
        elif self.index_12-1 < index <=self.index_21-1:
            cur_use_attri = '21'
            index = index - (self.index_12)#+1)
        elif self.index_21-1 < index <=self.index_22-1:
            cur_use_attri = '22'
            index = index - (self.index_21)#+1)
        
        img = cv2.imread(self.images_path[cur_use_attri][index]) #cv2.imread(self.images_path[cur_use_attri][index]) self.images[cur_use_attri][index]
        img = cv2.resize(img, (INPUT_SIZE[0], INPUT_SIZE[1]), interpolation = cv2.INTER_AREA)
        img = img.transpose(2,0,1)
        target = (self.attributes[cur_use_attri][index] - self.attributes_mean[use_uniform_mean])/self.attributes_std[use_uniform_mean]
        return torch.tensor(img).float(), torch.tensor(target).float(), cur_use_attri,  self.patient_idx[cur_use_attri][index]
        """
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        """
        
        

    def __len__(self):
        #use_attribute = '12'
        return self.index_22





class tooth_dataset_train_test():
    def __init__(self, anno_path,test_id, is_train=True):
        self.anno_path = anno_path
        self.is_train = is_train
        self.attributes_mean = {}
        self.attributes_std = {}
        self.attributes = {'11':[],'12':[],'21':[],'22':[]}
        self.images_path = {'11':[],'12':[],'21':[],'22':[]}
        self.images = {'11':[],'12':[],'21':[],'22':[]}
        self.patient_idx = {'11':[],'12':[],'21':[],'22':[]}
        self.need_attributes_idx = need_attributes_idx
        self.num_of_need_attri = len(self.need_attributes_idx)
        r = csv.reader(open(self.anno_path,  encoding='utf-8'))
        
        self.num_teeth = 0 
        for line in r:
            train_flag = line[2]
            if  test_id in train_flag:
                continue
            tooth_id = str(line[3])
            tooth_path = line[1]
            cur_attri = []
            lack = False
            for idx in self.need_attributes_idx:
                if len(line[idx])>0 and is_number(line[idx]):
                    cur_attri.append(float(line[idx]))
                else:
                    lack = True
            if lack:
                continue
            cur_attri = np.array(cur_attri)
            assert len(cur_attri) == self.num_of_need_attri
            self.images_path[tooth_id].append(tooth_path)
            self.patient_idx[tooth_id].append(line[0])
            #self.images[tooth_id].append(cv2.imread(tooth_path))
            self.attributes[tooth_id].append(cur_attri)
            self.num_teeth+=1
        
        print(" ")
        print("trainset?",is_train)
        print("total valid tooth",self.num_teeth)
        print("11",len(self.images_path['11']))
        print("12",len(self.images_path['12']))
        print("21",len(self.images_path['21']))
        print("22",len(self.images_path['22']))
        if '11' in use_attribute:
            self.index_11 = len(self.images_path['11'])
        else:
            self.index_11 = 0 
        if '12' in use_attribute:    
            self.index_12 = len(self.images_path['12'])+self.index_11
        else:
            self.index_12 = self.index_11
        if '21' in use_attribute: 
            self.index_21 = len(self.images_path['21'])+self.index_12
        else:
            self.index_21 = self.index_12
        if '22' in use_attribute: 
            self.index_22 = len(self.images_path['22'])+self.index_21
        else:
            self.index_22 = self.index_21
        print("self.index_11",self.index_11)
        print("self.index_12",self.index_12)
        print("self.index_21",self.index_21)
        print("self.index_22",self.index_22)
        
        
        for key in ['11','12','21','22']:
            print("key",key)
            matrix = np.array(self.attributes[key])
            print("matrix shape",matrix.shape)
            std = np.std(matrix,axis=0)
            mean = np.mean(matrix,axis=0)
            print('std',std)
            print('mean',mean)
            self.attributes_mean[key]= mean
            self.attributes_std[key]= std
    

    def __getitem__(self, index):
        #use_attribute = '12'
        if index <= self.index_11-1:
            cur_use_attri = '11'  
            index = index #- 1      
        elif self.index_11-1 < index <=self.index_12-1:
            cur_use_attri = '12'
            index = index - (self.index_11)#+1)
        elif self.index_12-1 < index <=self.index_21-1:
            cur_use_attri = '21'
            index = index - (self.index_12)#+1)
        elif self.index_21-1 < index <=self.index_22-1:
            cur_use_attri = '22'
            index = index - (self.index_21)#+1)
        #print('cur_use_attri',cur_use_attri)
        #print('ind',index)
        #print(len(self.images[cur_use_attri]))
        img = cv2.imread(self.images_path[cur_use_attri][index]) #cv2.imread(self.images_path[cur_use_attri][index])
        img = cv2.resize(img, (INPUT_SIZE[0], INPUT_SIZE[1]), interpolation = cv2.INTER_AREA)
        img = img.transpose(2,0,1)
        target = (self.attributes[cur_use_attri][index] - self.attributes_mean[use_uniform_mean])/self.attributes_std[use_uniform_mean]
        return torch.tensor(img).float(), torch.tensor(target).float(),cur_use_attri, self.patient_idx[cur_use_attri][index]
        """
        if self.is_train:
            img, target = self.train_img[index], self.train_label[index]
            if len(img.shape) == 2:
                img = np.stack([img] * 3, 2)
            img = Image.fromarray(img, mode='RGB')
            img = transforms.Resize((600, 600), Image.BILINEAR)(img)
            img = transforms.RandomCrop(INPUT_SIZE)(img)
            img = transforms.RandomHorizontalFlip()(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
        """
        
        

    def __len__(self):
        #use_attribute = '12'
        return self.index_22





if __name__ == '__main__':
    trainset = tooth_dataset_train(anno_path="/data2/xdw/teeth_annotation_small_output.csv")
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=1, drop_last=False)
    for i, data in enumerate(trainloader):
        print(data[0].shape, data[1])
    dataset = tooth_dataset_test(anno_path="/data2/xdw/teeth_annotation_small_output.csv")   
    for data in dataset:
        print(data[0].shape, data[1])
   
