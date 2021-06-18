# coding=gbk
import os
import shutil
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir_test, max_epoch, need_attributes_idx,use_uniform_mean,test_anno_csv_path, use_gpu, load_model_path,save_name
from core import model, dataset,resnet
from core.utils import init_log, progress_bar
import pandas as pd
from IPython import embed
import time


#os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu
start_epoch = 0


import torch
import argparse
import cv2
import numpy as np
import torch
from torch.autograd import Function
from torchvision import models, transforms

import os
import numpy as np
import shutil
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir, max_epoch, need_attributes_idx,use_uniform_mean,anno_csv_path, use_gpu, save_name, model_size, predtrain
from core import model, dataset,resnet
from core.utils import init_log, progress_bar
import pandas as pd
from IPython import embed


import os
import shutil
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir_test, max_epoch, need_attributes_idx,use_uniform_mean,test_anno_csv_path, use_gpu, load_model_path,save_name
import time

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            print("FeatureExtractor",name)
            x = module(x)
            print(x.requires_grad)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                #x = module(x)
                target_activations, x = self.feature_extractor(x)
                
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                print("modelout",name)
                print("in shape",x.shape)
                x = module(x)
                print("out shape",x.shape)
                print(x.requires_grad)
                #x.requires_grad = True

        return target_activations, x

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        #embed()
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.abs(cam)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        print("max",np.max(cam))
        cam = cam / np.max(cam)
        return cam,output


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)










file_dir_test = os.path.join(file_dir_test, save_name)
if not os.path.exists(file_dir_test):
    os.makedirs(file_dir_test)



# read dataset
testset = dataset.tooth_dataset_test(anno_path=test_anno_csv_path)
testloader = torch.utils.data.DataLoader(testset)
# define model
num_of_need_attri = len(need_attributes_idx)
net = resnet.resnet101(pretrained=False, num_classes = num_of_need_attri )
if load_model_path:
    ckpt = torch.load(load_model_path)
    for name in list(ckpt.keys()):
        ckpt[name.replace('module.','')] = ckpt[name]
        del ckpt[name]
    net.load_state_dict(ckpt)
    


# define optimizers
raw_parameters = list(net.parameters())

use_cuda = False

#net = net.cuda()
#net = DataParallel(net)
grad_cam = GradCam(model=net, feature_module=net.layer4, \
                       target_layer_names=["2"], use_cuda=use_cuda)

average_loss = [[111.1,111.1,111.1,111.1]]
head=['cur_use_attri','teeth_place']
for pre_name in ['target','output']:
    for attr_id in need_attributes_idx:
        head.append(pre_name+'_'+str(attr_id))
print(head)
#save_name =  'part6_dec4'#str(datetime.now().strftime('%Y%m%d_%H%M%S')) 

test_loss = 0
test_ori_loss = 0 
test_num = 0 
net.eval()
output_csv = []
total_time = 0
images_dir = '/data/shimr/teeth/'

for i, data in enumerate(testloader):
    #with torch.no_grad():
        #img, target = data[0].cuda(), data[1].cuda()    
        img, target = data[0], data[1]
        cur_use_attri, index = data[2],data[3]
        #embed()         
        batch_size = img.size(0)
        #print('test batch size',batch_size)#bs=1
        test_num += batch_size
        start = time.time()
        
        output= net(img)
        end = time.time()
        total_time += (end-start)
        
        target_unnorm = (target.detach().cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
        output_unnorm = (output.detach().cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
        cur_row =[]
        cur_row.append(str(cur_use_attri[0]))#.item()))
        cur_row.append(str(index))
        print('target_unnorm',target_unnorm)
        print('output_unnorm',output_unnorm)
        for tar in target_unnorm.reshape(-1):
            #print('t',tar)
            cur_row.append(str(tar))
        for out in output_unnorm.reshape(-1) :
            cur_row.append(str(out))
       
        ori_delta = (output-target).detach().abs().cpu().numpy()
        unnorm_delta = ori_delta * testset.attributes_std[use_uniform_mean]
        
        output_csv.append(cur_row)
        
        target_category = 1 #None
        grayscale_cam, output_cam = grad_cam(img, target_category)
        print("net output",output.detach().cpu().numpy())
        print("cam output",output_cam.detach().cpu().numpy())
        #embed()
        
        grayscale_cam = cv2.resize(grayscale_cam, (img.shape[2], img.shape[3]))
        #img = img.reshape(img.shape[1],img.shape[2],img.shape[3])
        cam = show_cam_on_image(img.reshape(img.shape[2],img.shape[3],img.shape[1]), grayscale_cam)
    
        #gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
        #gb = gb_model(img, target_category=target_category)
        #gb = gb.transpose((1, 2, 0))
    
        cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        #cam_gb = deprocess_image(cam_mask*gb)
        #gb = deprocess_image(gb)
        write_path = os.path.join(images_dir,str(index.item()).zfill(3),str(cur_use_attri[0]))
        print(write_path+"cam.jpg")
        cv2.imwrite(write_path+"cam.jpg", cam)
        #cv2.imwrite(write_path+'gb.jpg', gb)
        #cv2.imwrite(write_path+'cam_gb.jpg', cam_gb)
        
        
output_csv.insert(0,[str(total_time),str(test_num),str(total_time/test_num)])
loss_csv=pd.DataFrame(columns=head,data=output_csv)
loss_csv.to_csv(file_dir_test+'/{}_test_dataset.csv'.format(save_name),encoding='gbk')
