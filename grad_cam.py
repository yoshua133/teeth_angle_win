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
from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir, max_epoch, need_attributes_idx,use_uniform_mean,anno_csv_path, use_gpu, save_name, model_size, pretrain


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
            #print("FeatureExtractor",name)
            x = module(x)
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
                #print("modelout",name)
                #print("in shape",x.shape)
                x = module(x)
                #print("out shape",x.shape)

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
    heatmap = np.float32(heatmap) #/ 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)
def save_grey(mask,path):
    heatmap = np.uint8(255.0 * mask)
    img = plt.imsave(path, heatmap)
    #img.save(path)
    

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.fc_weight = self.model.fc.weight
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.model.zero_grad()
        self.feature_module.zero_grad()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)
        output_model, last_layer_feature = self.model(input_img)
        last_layer_feature= torch.squeeze(last_layer_feature, 0)
        #embed()
        print("output_model",output_model)
        print("output",output)
        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()      
        one_hot = torch.sum(one_hot * output)  
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]
        last_layer_feature = last_layer_feature.cpu().data.numpy()
        fc_weight = self.fc_weight[target_category,:].cpu().data.numpy()
        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        # fcweight=10-3~-4, weight=10-4~-5
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        fc_cam = np.zeros(target.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
            fc_cam += fc_weight[i,]  * last_layer_feature[i,:,:]
        #embed()
        print("fc_cam max",np.max(fc_cam))
        print("fc_cam min",np.min(fc_cam))
        print("fc_cam mean",np.mean(fc_cam))
        cam = cam
        fc_cam = np.abs(fc_cam)
        
        cam = cv2.resize(cam, input_img.shape[2:])
        r = np.max(cam) - np.min(cam)
        cam = cam - np.min(cam)
        cam = cam / r
        
        fc_cam = cv2.resize(fc_cam, input_img.shape[2:])
        r = np.max(fc_cam) - np.min(fc_cam)
        fc_cam = fc_cam - np.min(fc_cam)
        fc_cam = fc_cam / r
        
        return cam,fc_cam


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

        output,features = self.forward(input_img)

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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)
from IPython import embed

def write_output(image_path, output_path, grad_cam, model, use_cuda, target_category,gb_model):
    img = cv2.imread(image_path)#, 1)
    img = np.float32(img) #/ 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    #embed()
    print("after resize image shape",img.shape)
    img_c = img.copy().transpose(2,0,1)
    input_img = torch.tensor(img_c).unsqueeze(0)#preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 0 #None
    grayscale_cam, fc_cam = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    fc_cam = cv2.resize(fc_cam, (img.shape[1], img.shape[0]))
    
    save_grey(grayscale_cam,'grey_cam.png')
    save_grey(fc_cam,'grey_fc_cam.png')
    
    cam = show_cam_on_image(img, grayscale_cam)
    fc_cam = show_cam_on_image(img, fc_cam)

    
    gb = gb_model(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite(output_path + "cam.jpg", cam)
    #cv2.imwrite(output_path + "fc_cam.jpg", fc_cam)
    cv2.imwrite(output_path + 'gb.jpg', gb)
    #cv2.imwrite(output_path + 'cam_gb.jpg', cam_gb)

if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    #args = get_args()
    use_cuda = False
    num_of_need_attri = len(need_attributes_idx)
    test_id =0 
    num_of_need_attri = 3 #len(need_attributes_idx)
    
    
    
    model = resnet.resnet101(pretrained=False,num_classes = num_of_need_attri)
    
    model.eval()
    load_model_path = os.path.join(save_dir, '20210219_171456part0_Feb19res101','model_param.pkl')
    if load_model_path:
        ckpt = torch.load(load_model_path)
        for name in list(ckpt.keys()):
            ckpt[name.replace('module.','')] = ckpt[name]
            del ckpt[name]
        model.load_state_dict(ckpt)
    grad_cam = GradCam(model=model, feature_module=model.layer4, \
                       target_layer_names=["2"], use_cuda=use_cuda)
    #embed()
    """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    img = img.transpose(2,0,1)
    """
    images_dir = '/data/shimr/teeth/'
    target_category = 1
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    if False:
        for patient_id in os.listdir(images_dir):
            patient_dir = os.path.join(images_dir,patient_id)
            for tooth_id in os.listdir(patient_dir):
                if tooth_id.startswith('crop') and tooth_id.endswith('tif'):
                    image_path = os.path.join(patient_dir,tooth_id)
                    print(image_path)
                    output_path = image_path.replace('.tif','')
                    write_output(image_path, output_path, grad_cam, model, use_cuda, target_category,gb_model)
    
    num_id = '015'
    image_path = "/data/shimr/teeth/204/cropped_image204,22 Maxilla, Application.tif"
    output_path = "/data/shimr/visual/"
    #print(os.listdir(image_path))
    #for p in os.listdir(image_path):
    #    if p.startswith('crop'):
    #        image_path = os.path.join(image_path,p)
    #        break 
    write_output(image_path, output_path, grad_cam, model, use_cuda, target_category,gb_model)
    
    """
    img = cv2.imread(image_path, 1)
    img = np.float32(img) / 255
    # Opencv loads as BGR:
    img = img[:, :, ::-1]
    img = cv2.resize(img, (224,224), interpolation = cv2.INTER_AREA)
    
    print("after resize image shape",img.shape)
    #img = img.transpose(2,0,1)
    input_img = preprocess_image(img)

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested category.
    target_category = 1 #None
    grayscale_cam,fc = grad_cam(input_img, target_category)

    grayscale_cam = cv2.resize(grayscale_cam, (img.shape[1], img.shape[0]))
    cam = show_cam_on_image(img, grayscale_cam)

    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    gb = gb_model(input_img, target_category=target_category)
    gb = gb.transpose((1, 2, 0))

    cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)

    cv2.imwrite("cam.jpg", cam)
    cv2.imwrite('gb.jpg', gb)
    cv2.imwrite('cam_gb.jpg', cam_gb)
    """