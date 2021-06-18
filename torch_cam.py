import torch
from torchcam.cams import *

from torchvision.models import resnet18

import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms
import torchvision
import os
import numpy as np
import shutil
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR



from config import BATCH_SIZE,  SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir_test, max_epoch, need_attributes_idx,use_uniform_mean,test_anno_csv_path, use_gpu, load_model_path,test_save_name,anno_csv_path,   model_size, pretrain, bigger, model_name,load_file, load_time


from core import model, dataset,resnet
from core.utils import init_log, progress_bar
import pandas as pd
from IPython import embed
import matplotlib.pyplot as plt

import os
import shutil
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR

import time



def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) #/ 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
def save_grey(mask,path):
    heatmap = np.uint8(255.0 * mask)
    img = plt.imsave(path, heatmap)
    #img.save(path)
    
def max_norm(cam):
    r = np.max(cam) - np.min(cam)
    cam = cam - np.min(cam)
    cam = cam / r
    return cam
    
    
    

model_name = 'resnext101_32x8d'
model_size = '101'
use_cuda = False
need_attributes_idx = [7,8]
num_of_need_attri = len(need_attributes_idx)
test_id =0 
#num_of_need_attri = 3 #len(need_attributes_idx)
#load_model_path_i = os.path.join(save_dir, load_time+load_file+'_'+str(test_id),'model_param.pkl') 

if model_name == 'resnet':
    if model_size == '50':
            net = resnet.resnet50(pretrained=pretrain, num_classes = num_of_need_attri,bigger=bigger )           
    elif model_size == '34':
        net = resnet.resnet34(pretrained=pretrain, num_classes = num_of_need_attri )
    elif model_size == '101':
        net = resnet.resnet50(pretrained=pretrain, num_classes = num_of_need_attri,bigger=bigger )
    elif model_size == '152':
        net = resnet.resnet152(pretrained=pretrain, num_classes = num_of_need_attri )        
elif model_name == 'vgg':
    if model_size == '11':
        net = torchvision.models.vgg11_bn(pretrained=pretrain, num_classes = num_of_need_attri )
    elif model_size == '16':
        net = torchvision.models.vgg16_bn(pretrained=pretrain, num_classes = num_of_need_attri )
    elif model_size == '16_nobn':
        net = torchvision.models.vgg16(pretrained=pretrain, num_classes = num_of_need_attri )
    elif model_size == '19':
        net = torchvision.models.vgg19_bn(pretrained=pretrain, num_classes = num_of_need_attri )
        
elif model_name == "resnext101_32x8d":
    net = torchvision.models.resnext101_32x8d(pretrained=pretrain, num_classes = num_of_need_attri )

elif model_name == "inception_v3":
    net = torchvision.models.inception_v3(pretrained=pretrain, num_classes = num_of_need_attri, aux_logits =False )
    
model = net
model = model.to(device = 'cpu')
model.eval()
load_model_path_i = os.path.join(save_dir, '20210418_164709part0_apr18_revised_crop_1_725_train_aug_only_2_p_0.2_78resnext101_32x8d_101pretrain-False','model_param.pkl') #20210511_142559kfold_may5_revised_crop1_725_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_1
#embed()
if load_model_path_i:
    ckpt = torch.load(load_model_path_i,map_location={'cuda:0': 'cuda:7'})
    #embed()
    for name in list(ckpt.keys()):
        ckpt[name.replace('module.','')] = ckpt[name]
        del ckpt[name]
    model.load_state_dict(ckpt)


"""
# Hook your model before the forward pass
cam_extractor = SmoothGradCAMpp(model,"layer4","conv1")#,"fc") #SmoothGradCAMpp  ScoreCAM
#GradCAM GradCAMpp
for patient_id in range(400,500):
    patient_id = str(patient_id).zfill(3)
    for tooth_id in ['11','12','21','22']:
            img_dir = "/data/shimr/teeth/{}/".format(patient_id)
            for file_name in os.listdir(img_dir):
                if not (file_name.endswith('tif') and file_name.startswith('cropped_image{}'.format(patient_id)) and ',' in file_name and  tooth_id in file_name.split(',')[1]):
                    continue
                #img_name = "/data/shimr/teeth/{}/cropped_image{}, {}Maxilla,Application.tif".format(patiend_id,patiend_id,tooth_id)
                img_name = os.path.join(img_dir,file_name)
                print(patient_id,tooth_id,img_name)
                if not os.path.exists(img_name):
                    continue
                image_path =  os.path.join("/data/shimr/teeth/",patient_id,img_name)
                output_path = os.path.join("/data/shimr/visual/",patient_id, tooth_id)#/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/cam_imgae     os.path.join("/data/shimr/visual/",patient_id, tooth_id)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    
                img = cv2.imread(image_path)#, 1)
                img = np.float32(img) #/ 255
                #print("target_category",target_category)
                # Opencv loads as BGR:
                img = img[:, :, ::-1]
                img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                #
                
                img_c = img.copy().transpose(2,0,1)
                input_img = torch.tensor(img_c)[None,:,:,:]#.unsqueeze(0)#preprocess_image(img)
                print("after resize image shape",input_img.shape)
                # By default the last conv layer will be selected
                cam_extractor._hooks_enabled = True
                model.zero_grad()
                out = model(input_img)
                print(out.squeeze(0).argmax().item())
                # Retrieve the CAM
                for target_category in [0,1]:
                    activation_map = cam_extractor(target_category, out).detach().cpu().numpy()
                    activation_map_img = cv2.resize(activation_map, (img.shape[1], img.shape[0]))
                    activation_map_img = show_cam_on_image(img, activation_map_img)
                    cv2.imwrite(os.path.join(output_path,str(target_category)+"SmoothGradCAMpp.jpg"),activation_map_img)
                    #embed()
                cam_extractor.clear_hooks()
                cam_extractor._hooks_enabled = False


"""


# Hook your model before the forward pass
cam_extractor = GradCAM(model,"layer4")#,"fc") #SmoothGradCAMpp  ScoreCAM
#GradCAM GradCAMpp
for patient_id in range(400,500):
    patient_id = str(patient_id).zfill(3)
    for tooth_id in ['11','12','21','22']:
            img_dir = "/data/shimr/teeth/{}/".format(patient_id)
            for file_name in os.listdir(img_dir):
                if not (file_name.endswith('tif') and file_name.startswith('cropped_image{}'.format(patient_id)) and ',' in file_name and  tooth_id in file_name.split(',')[1]):
                    continue
                #img_name = "/data/shimr/teeth/{}/cropped_image{}, {}Maxilla,Application.tif".format(patiend_id,patiend_id,tooth_id)
                img_name = os.path.join(img_dir,file_name)
                print(patient_id,tooth_id,img_name)
                if not os.path.exists(img_name):
                    continue
                image_path =  os.path.join("/data/shimr/teeth/",patient_id,img_name)
                output_path = os.path.join("/data/shimr/visual/",patient_id, tooth_id)#/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/cam_imgae     os.path.join("/data/shimr/visual/",patient_id, tooth_id)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    
                img = cv2.imread(image_path)#, 1)
                img = np.float32(img) #/ 255
                #print("target_category",target_category)
                # Opencv loads as BGR:
                img = img[:, :, ::-1]
                img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                #
                
                img_c = img.copy().transpose(2,0,1)
                input_img = torch.tensor(img_c)[None,:,:,:]#.unsqueeze(0)#preprocess_image(img)
                print("after resize image shape",input_img.shape)
                # By default the last conv layer will be selected
                cam_extractor._hooks_enabled = True
                model.zero_grad()
                out = model(input_img)
                print(out.squeeze(0).argmax().item())
                # Retrieve the CAM
                for target_category in [0,1]:
                    activation_map = cam_extractor(target_category, out).detach().cpu().numpy()
                    activation_map_img = cv2.resize(activation_map, (img.shape[1], img.shape[0]))
                    activation_map_img = show_cam_on_image(img, activation_map_img)
                    cv2.imwrite(os.path.join(output_path,str(target_category)+"GradCAM.jpg"),activation_map_img)
                    #embed()
                cam_extractor.clear_hooks()
                cam_extractor._hooks_enabled = False



cam_extractor = GradCAMpp(model,"layer4")#,"fc") #SmoothGradCAMpp  ScoreCAM
#GradCAM GradCAMpp
for patient_id in range(400,500):
    patient_id = str(patient_id).zfill(3)
    for tooth_id in ['11','12','21','22']:
            img_dir = "/data/shimr/teeth/{}/".format(patient_id)
            for file_name in os.listdir(img_dir):
                if not (file_name.endswith('tif') and file_name.startswith('cropped_image{}'.format(patient_id)) and ',' in file_name and  tooth_id in file_name.split(',')[1]):
                    continue
                #img_name = "/data/shimr/teeth/{}/cropped_image{}, {}Maxilla,Application.tif".format(patiend_id,patiend_id,tooth_id)
                img_name = os.path.join(img_dir,file_name)
                print(patient_id,tooth_id,img_name)
                if not os.path.exists(img_name):
                    continue
                image_path =  os.path.join("/data/shimr/teeth/",patient_id,img_name)
                output_path = os.path.join("/data/shimr/visual/",patient_id, tooth_id)#/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/cam_imgae     os.path.join("/data/shimr/visual/",patient_id, tooth_id)
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                    
                img = cv2.imread(image_path)#, 1)
                img = np.float32(img) #/ 255
                #print("target_category",target_category)
                # Opencv loads as BGR:
                img = img[:, :, ::-1]
                img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
                #
                
                img_c = img.copy().transpose(2,0,1)
                input_img = torch.tensor(img_c)[None,:,:,:]#.unsqueeze(0)#preprocess_image(img)
                print("after resize image shape",input_img.shape)
                # By default the last conv layer will be selected
                cam_extractor._hooks_enabled = True
                model.zero_grad()
                out = model(input_img)
                print(out.squeeze(0).argmax().item())
                # Retrieve the CAM
                for target_category in [0,1]:
                    activation_map = cam_extractor(target_category, out).detach().cpu().numpy()
                    activation_map_img = cv2.resize(activation_map, (img.shape[1], img.shape[0]))
                    activation_map_img = show_cam_on_image(img, activation_map_img)
                    cv2.imwrite(os.path.join(output_path,str(target_category)+"GradCAMpp.jpg"),activation_map_img)
                    #embed()
                cam_extractor.clear_hooks()
                cam_extractor._hooks_enabled = False



