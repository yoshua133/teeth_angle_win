# coding=gbk
import os
import shutil
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE,  SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir_test, max_epoch, need_attributes_idx,use_uniform_mean,test_anno_csv_path, use_gpu, load_model_path,test_save_name,anno_csv_path,   model_size, pretrain, bigger, model_name,load_file, load_time
from core import  dataset,resnet
from core.utils import init_log, progress_bar
import pandas as pd
from IPython import embed
import time
import numpy as np
import torchvision


os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu
start_epoch = 0

for test_id in range(6):
    
    load_model_path_i = os.path.join(save_dir, load_time+load_file+'_'+str(test_id),'model_param.pkl') 
    if test_id ==5:
        load_model_path_i = os.path.join(save_dir, load_time+load_file,'model_param.pkl') 
    #if not os.path.exists(load_model_path_i):
    #    for dir_name in os.listdir(save_dir):
    #        if load_file
    if not os.path.exists(load_model_path_i):
        continue
    #file_dir_test = os.path.join(file_dir_test, test_save_name)+'_'+str(test_id)
    save_csv_path_test = file_dir_test+'/{}_test_dataset_{}.csv'.format(test_save_name,test_id)
    save_csv_path_train = file_dir_test+'/{}_train_dataset_{}.csv'.format(test_save_name,test_id)
    
    if not os.path.exists(file_dir_test):
        os.makedirs(file_dir_test)
    if test_id <5:
        trainset = dataset.tooth_dataset_train_test(anno_path=test_anno_csv_path,test_id = str(test_id))
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False)
        
        # read dataset
        testset = dataset.tooth_dataset_test(anno_path=test_anno_csv_path,test_id = str(test_id))
    else:
        trainset = dataset.tooth_dataset_train_test(anno_path=test_anno_csv_path,test_id = "test")
        trainloader = torch.utils.data.DataLoader(trainset, shuffle=False)
        
        # read dataset
        testset = dataset.tooth_dataset_test(anno_path=test_anno_csv_path,test_id = "test")
    print("test mean",testset.attributes_mean)
    print("test std",testset.attributes_std)
    print("train mean",trainset.attributes_mean)
    print("train std",trainset.attributes_std)
    testset.attributes_mean = trainset.attributes_mean
    testset.attributes_std = trainset.attributes_std
    
    testloader = torch.utils.data.DataLoader(testset, shuffle=False)
    # define model
    num_of_need_attri = len(need_attributes_idx)
    
    print("model",model_name)
    print(model_size)
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
    
    if load_model_path_i:
        ckpt = torch.load(load_model_path_i)
        for name in list(ckpt.keys()):
            ckpt[name.replace('module.','')] = ckpt[name]
            del ckpt[name]
        net.load_state_dict(ckpt)
        
    
    
    # define optimizers
    raw_parameters = list(net.parameters())
    
    
    
    net = net.cuda()
    net = DataParallel(net)
    
    
    
    head=['cur_use_attri','teeth_place']
    for pre_name in ['target','output']:
        for attr_id in need_attributes_idx:
            head.append(pre_name+'_'+str(attr_id))
    if len(need_attributes_idx)==2:
        use_9 = True
    else:
        use_9 = False
    if use_9:
        head.append('target_9')
        head.append('output_9')
    print(head)
    #test_save_name =  'part6_dec4'#str(datetime.now().strftime('%Y%m%d_%H%M%S')) 
    
    
    average_loss = [[111.1,111.1,111.1,111.1]]
    test_loss = 0
    test_ori_loss = 0 
    test_num = 0 
    net.eval()
    output_csv = []
    total_time = 0
    seg_dict = {1:0,2:0,5:0,10:0}
    mae_total = 0
    for i, data in enumerate(testloader):
        with torch.no_grad():
            img, target = data[0].cuda(), data[1].cuda()    
            cur_use_attri, index = data[2],data[3]
            #embed()         
            batch_size = img.size(0)
            #print('test batch size',batch_size)#bs=1
            test_num += batch_size
            start = time.time()
            output = net(img) #feature
            end = time.time()
            total_time += (end-start)
            target_unnorm = (target.cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
            output_unnorm = (output.cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
            target_unnorm = target_unnorm.reshape(-1)
            output_unnorm = output_unnorm.reshape(-1)
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
            if use_9:
                target_9 =  target_unnorm[0] - target_unnorm[1]
                output_9 =  output_unnorm[0] - output_unnorm[1]
                cur_row.append(str(target_9))
                cur_row.append(str(output_9))
            
            ori_delta = (output-target).abs().cpu().numpy()
            if use_9:
                unnorm_delta = np.abs(target_9-output_9) #
            else:
                unnorm_delta = ori_delta * testset.attributes_std[use_uniform_mean]
            mae_total = mae_total + unnorm_delta
            if np.mean(unnorm_delta)<=1 :
                seg_dict[1] +=1
            elif np.mean(unnorm_delta) <=2.5:
                seg_dict[2] +=1
            elif np.mean(unnorm_delta) <=5:
                seg_dict[5] +=1
            elif np.mean(unnorm_delta) <=10:
                seg_dict[10] +=1
            #embed()
            output_csv.append(cur_row)
            
    output_csv.insert(0,[str(total_time),str(test_num),str(total_time/test_num)])
    mae_print = list()
    if use_9:
        mae_print.append(str(mae_total/test_num))
    else:
        for i in range(mae_total.shape[0]):
            mae_print.append(str(mae_total[i]/test_num))
    output_csv.insert(0,mae_print)
    output_csv.insert(0,["0~1",str(seg_dict[1]/test_num)])
    output_csv.insert(0,["1~2.5",str(seg_dict[2]/test_num)])
    output_csv.insert(0,["2.5~5",str(seg_dict[5]/test_num)])
    output_csv.insert(0,["5~10",str(seg_dict[10]/test_num)])
    #embed()
    loss_csv=pd.DataFrame(columns=head,data=output_csv)
    #embed()
    loss_csv.to_csv(save_csv_path_test,encoding='gbk')
    
    
    
    
    average_loss = [[111.1,111.1,111.1,111.1]]
    test_loss = 0
    test_ori_loss = 0 
    test_num = 0 
    net.eval()
    output_csv = []
    total_time = 0
    """
    for i, data in enumerate(trainloader):
        with torch.no_grad():
            img, target = data[0].cuda(), data[1].cuda()    
            cur_use_attri, index = data[2],data[3]
            #embed()         
            batch_size = img.size(0)
            #print('test batch size',batch_size)#bs=1
            test_num += batch_size
            start = time.time()
            output= net(img) # , features
            end = time.time()
            total_time += (end-start)
            target_unnorm = (target.cpu().numpy()* trainset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
            output_unnorm = (output.cpu().numpy()* trainset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
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
            
                
            ori_delta = (output-target).abs().cpu().numpy()
            unnorm_delta = ori_delta * testset.attributes_std[use_uniform_mean]
            
            output_csv.append(cur_row)
            
    output_csv.insert(0,[str(total_time),str(test_num),str(total_time/test_num)])
    loss_csv=pd.DataFrame(columns=head,data=output_csv)
    loss_csv.to_csv(save_csv_path_train,encoding='gbk')
    """

