

#os.environ['CUDA_VISIBLE_DEVICES'] = use_gpu


if __name__ ==  '__main__':
    import os
    import numpy as np
    import shutil
    import torch.utils.data
    from torch.nn import DataParallel
    from datetime import datetime
    from torch.optim.lr_scheduler import MultiStepLR
    from config import BATCH_SIZE, PROPOSAL_NUM, SAVE_FREQ, LR, WD, resume, save_dir,use_attribute, file_dir, max_epoch, need_attributes_idx,use_uniform_mean,anno_csv_path, use_gpu, save_name, model_size, pretrain,loss_weight_mask_thres, model_name, bigger, start_from_test_id, test_save_name,file_dir_test,time_first,loss_name
    from core import  dataset,resnet
    from core.utils import init_log, progress_bar
    import pandas as pd
    import torchvision.models  
    from IPython import embed
    import time
    
    
    start_epoch = 0
    num_of_need_attri = len(need_attributes_idx)
    print("use attribute",need_attributes_idx)
    print("cuda available", torch.cuda.is_available())
    print("start training")



    save_dir_ori = save_dir
    file_dir_ori = file_dir
    #time_first =  datetime.now().strftime('%Y%m%d_%H%M%S')
    former_best = list()
    #for test_id in range(5):
    test_id = start_from_test_id
    #if start_from_test_id>test_id:
    #    continue
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
    
    elif model_name == "resnext50_32x4d":
        net = torchvision.models.resnext50_32x4d(pretrained=pretrain, num_classes = num_of_need_attri )
    
    elif model_name == "inception_v3":
        net = torchvision.models.inception_v3(pretrained=pretrain, num_classes = num_of_need_attri, aux_logits =False )
        
    elif model_name == "wide_resnet101_2":
        net = torchvision.models.wide_resnet101_2(pretrained=pretrain, num_classes = num_of_need_attri)
    elif model_name == "densenet":
        net = torchvision.models.densenet201(pretrained=pretrain, num_classes = num_of_need_attri)
    
    
    
        
    save_dir = os.path.join(save_dir_ori,time_first+save_name+"_{}".format(test_id))
    file_dir = os.path.join(file_dir_ori, time_first+save_name+"_{}".format(test_id))
    if not os.path.exists(save_dir):
      os.makedirs(save_dir)
    if not os.path.exists(file_dir_test):
        os.makedirs(file_dir_test)
        
    if loss_name == "L1":
        creterion = torch.nn.L1Loss()
    elif loss_name == "L2":
        creterion = torch.nn.MSELoss()
    elif loss_name == "smooth_L1":
        creterion = torch.nn.SmoothL1Loss()
    elif loss_name == "huber":
        creterion = torch.nn.HuberLoss()
    # read dataset
    trainset = dataset.tooth_dataset_train(anno_path=anno_csv_path,test_id = test_id)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                              shuffle=True, num_workers=8, drop_last=False, pin_memory= False)
    testset = dataset.tooth_dataset_test(anno_path=anno_csv_path,test_id = test_id)
    testset.attributes_mean = trainset.attributes_mean
    testset.attributes_std = trainset.attributes_std
    print("test mean",testset.attributes_mean)
    print("test std",testset.attributes_std)
    testloader = torch.utils.data.DataLoader(testset, pin_memory= False)
    # define model
    
    
        
    #embed()
    if resume :
        ckpt = torch.load(resume)
        for name in list(ckpt.keys()):
          ckpt[name.replace('module.','')] = ckpt[name]
          del ckpt[name]
        net.load_state_dict(ckpt)
        start_epoch = 0#ckpt['epoch'] + 1
    
    
    # define optimizers
    raw_parameters = list(net.parameters())
    
    
    raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
    #lr_schedule = optim.lr_schedule.StepLR(raw_optimizer,
    schedulers = [MultiStepLR(raw_optimizer, milestones=[160, 200], gamma=0.1)]
    net = net.cuda()
    net = DataParallel(net)
    
    average_loss = [[111.1,111.1,111.1,100.0,100.0]]
    average_loss.extend(former_best)
    head=['train_loss_unit_degree','train_ori_loss_unit_std','test_loss','test_ori_loss','target_loss']
    
    
    
    
    
    test_head=['cur_use_attri','teeth_place']
    for pre_name in ['target','output']:
        for attr_id in need_attributes_idx:
            test_head.append(pre_name+'_'+str(attr_id))
    if len(need_attributes_idx)==2:
        use_9 = True
    else:
        use_9 = False
    if use_9:
        test_head.append('target_9')
        test_head.append('output_9')
    print("test_head",test_head)
    #test_save_name =  'part6_dec4'#str(datetime.now().strftime('%Y%m%d_%H%M%S')) 
    
    save_csv_path_test = os.path.join(file_dir,'test_dataset_{}.csv'.format(test_id))#,test_save_name))
    #save_csv_path_train = file_dir_test+'/{}_train_dataset_{}.csv'.format(test_save_name,test_id)
    
    
    
    
    
    
    
    
    for epoch in range(start_epoch, max_epoch):
       
    
        # begin training
        print('--' * 50)
        net.train()
        train_num = 0
        train_loss = 0
        train_ori_loss = 0
        
        
        print("before train")
        for i, data in enumerate(trainloader):
            if i%50==0:
                print("in train",i)
            img, target = data[0].cuda(), data[1].cuda()
            batch_size = img.size(0)
            #print("batch size",batch_size)
            train_num += batch_size
            raw_optimizer.zero_grad()
            output = net(img)
           
            loss = torch.abs(output - target).reshape(-1) # cross entrophy
            weight = torch.ones(loss.shape[0],1).cuda()
            #weight[loss> torch.tensor(loss_weight_mask_thres/trainset.attributes_std[use_uniform_mean]).cuda().reshape(-1)] = 0.5
            #print("loss",loss)
            #print("weight",weight)
            loss = loss * weight
            loss = loss.sum()
            
            ori_delta = (output-target).abs().cpu().detach().numpy()
            ori_delta_mean = ori_delta.mean()
            if train_num %100 ==0 and np.random.random()<-0.1:
                print("target",target)
                print("outputs",output)
                print("loss",loss)
                print("unnorm delta",ori_delta * (trainset.attributes_std[use_uniform_mean]).reshape(-1))
                print("ori  delta",ori_delta)
    
            train_ori_loss += ori_delta_mean * batch_size
            unnorm_delta = ori_delta * (trainset.attributes_std[use_uniform_mean]).reshape(-1)
            train_loss += unnorm_delta.mean() * batch_size
            loss.backward()
            raw_optimizer.step()
            #progress_bar(i, len(trainloader), 'train')
        for scheduler in schedulers:
            scheduler.step()
        if not os.path.exists(save_dir):
                os.mkdir(save_dir)
        if not os.path.exists(file_dir):
                os.mkdir(file_dir)
        if epoch<1  :
            shutil.copy( 'config.py', file_dir+'/config.py')
            shutil.copy( 'train.py', file_dir+'/train.py')
            shutil.copy( 'core/dataset.py', file_dir+'/dataset.py')
            shutil.copy( 'core/resnet.py', file_dir+'/resnet.py')
        if epoch % 5 == 0 or epoch==1:
            test_loss = 0
            test_ori_loss = 0
            test_target_loss = 0
            test_num = 0
            net.eval()
            
            total_time = 0
            output_csv = []
            mae_total = 0
            mse_total = 0
            seg_dict = {1:0,2:0,5:0,10:0}
            
            for i, data in enumerate(testloader):
                with torch.no_grad():
                    img, target = data[0].cuda(), data[1].cuda()
                    cur_use_attri, index = data[2],data[3]
                    
                    batch_size = img.size(0)
                    #print('test batch size',batch_size)#bs=1
                    test_num += batch_size
                    raw_optimizer.zero_grad()
                    start = time.time()
                    output = net(img)
                    end = time.time()
                    total_time += (end-start)
                    # calculate loss
                    #print("target",target.shape)
                    #print("target type",type(target))
                    #print("outputs",output.shape)
                    #print("output type",type(output))
                    #print("loss",loss)
                    #loss = creterion(output, target)
                    ori_delta = (output-target).abs().cpu().numpy()
                    unnorm_delta = ori_delta * (trainset.attributes_std[use_uniform_mean]).reshape(-1)
                    unnorm_out = output.cpu().numpy() * (trainset.attributes_std[use_uniform_mean]).reshape(-1)
                    unnorm_tar = target.cpu().numpy() * (trainset.attributes_std[use_uniform_mean]).reshape(-1)
                    if unnorm_delta.shape[-1]==1:
                        test_target_loss += unnorm_delta.sum()
                    elif unnorm_delta.shape[-1]==3:
                        test_target_loss += unnorm_delta[:,-1].sum()
                    elif unnorm_delta.shape[-1]==2:
                        test_target_loss += np.abs( ( (unnorm_out[:,0]-unnorm_out[:,1]) - (unnorm_tar[:,0]-unnorm_tar[:,1]) ) ).sum()
                    else:
                        test_target_loss += unnorm_delta.sum()
                        
                    # from test_dataset.py
                    target_unnorm = (target.cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
                    output_unnorm = (output.cpu().numpy()* testset.attributes_std[use_uniform_mean])+testset.attributes_mean[use_uniform_mean]
                    target_unnorm = target_unnorm.reshape(-1)
                    output_unnorm = output_unnorm.reshape(-1)
                    cur_row =[]
                    cur_row.append(str(cur_use_attri[0]))#.item()))                                                                                                 
                    cur_row.append(str(index))
                    
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
                        
                    if use_9:
                        unnorm_delta = np.abs(target_9-output_9) #
                    else:
                        unnorm_delta = ori_delta * testset.attributes_std[use_uniform_mean]
                    mae_total = mae_total + unnorm_delta
                    mse_total = mse_total + unnorm_delta.reshape(-1) * unnorm_delta.reshape(-1)
                    
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
                    
                    """
                    if unnorm_delta[-1] <=1 :
                        seg_dict[1] +=1
                    elif unnorm_delta[-1] <=2.5:
                        seg_dict[2] +=1
                    elif unnorm_delta[-1] <=5:
                        seg_dict[5] +=1
                    elif unnorm_delta[-1] <=10:
                        seg_dict[10] +=1
                    """
                    #loss is the mean distance between two tensor
                    test_loss += unnorm_delta.mean()*batch_size
                    test_ori_loss += ori_delta.mean()*batch_size
                    # calculate accuracy
                    
            output_csv.insert(0,[str(total_time),str(test_num),str(total_time/test_num)])
            mae_print = list()
            if use_9:
                mae_print.append(str(mae_total/test_num))
            else:
                mae_print.append(" mae ")
                for i in range(mae_total.shape[0]):
                    mae_print.append(str(mae_total[i]/test_num))
                mae_print.append(" mse ")
                for i in range(mse_total.shape[0]):
                    mae_print.append(str(mse_total[i]/test_num))
            output_csv.insert(0,mae_print)
            output_csv.insert(0,["0~1",str(seg_dict[1]/test_num)])
            output_csv.insert(0,["1~2.5",str(seg_dict[2]/test_num)])
            output_csv.insert(0,["2.5~5",str(seg_dict[5]/test_num)])
            output_csv.insert(0,["5~10",str(seg_dict[10]/test_num)])
            #embed()
            loss_csv=pd.DataFrame(columns=test_head,data=output_csv)
            #embed()
            loss_csv.to_csv(save_csv_path_test,encoding='gbk')
    
    
    
    
    
            print("epoch:{} mean loss, L1 gap divided by std".format(epoch),test_loss/test_num,"  ori loss ",\
              test_ori_loss/test_num,"target loss", test_target_loss/test_num)
            print("test_num",test_num)
            #train_ori_loss = trainset.attributes_std[use_uniform_mean][0]*train_loss.item()/train_num
            #test_ori_loss = trainset.attributes_std[use_uniform_mean][0]*test_loss.item()/test_num
            average_loss.append([train_loss/train_num, train_ori_loss/train_num, test_loss/test_num, test_ori_loss/test_num, test_target_loss/test_num])
            if test_target_loss/test_num < average_loss[0][4]:
                average_loss[0] = [train_loss/train_num, train_ori_loss/train_num, test_loss/test_num, test_ori_loss/test_num, test_target_loss/test_num]
            loss_csv=pd.DataFrame(columns=head,data=average_loss)
            loss_csv.to_csv(file_dir+'/{}_loss.csv'.format(save_name),encoding='gbk')
            f = open(file_dir+'/{}_mean.txt'.format(save_name),'w')
            f.write(str(trainset.attributes_mean))
            f.close()
            f2 = open(file_dir+'/{}_std.txt'.format(save_name),'w')
            f2.write(str(trainset.attributes_std))
            f2.close()
            print("finish writing")
            net_state_dict = net.state_dict()
            torch.save(net_state_dict,save_dir+'/model_param.pkl')
            print("finish save")
    li = list()
    li.append(average_loss[0])
    former_best.extend(li)