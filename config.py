import os
from datetime import datetime
PROPOSAL_NUM = ''
CAT_NUM=''

BATCH_SIZE = 4
INPUT_SIZE = (224, 224)  # (w, h)
bigger = False

LR = 0.0001  # learning rate
WD = 1e-4   # weight decay 0.0001
SAVE_FREQ = 1
loss_weight_mask_thres = -1

use_attribute = ['11','12','21','22']


test_model = 'model.ckpt'
model_name = 'wide_resnet101_2'  #如果是inception的话 INPUT_SIZE要改成299,299
model_size = '101'
loss_name = 'L2' #根据train.py中的设定修改，在train.py的82行
pretrain = False
dataset_size = 725

flip_prob = 0  #数据扩增 翻转概率
crop_method = 1  #0代表整张x光范围, 1代表上牙齿范围

save_dir = 'data\\model_save\\' #保存模型的路径
file_dir = 'train_files\\'  #测试结果
time_first =  datetime.now().strftime('%Y%m%d_%H%M%S')

resume = "" # os.path.join(save_dir, '20210512_172826kfold_may5_revised_crop1_725_aug_p_0_attri_7_8_image_randomresnext101_32x8d_101pretrain-Falsesize224_0','model_param.pkl')
start_from_test_id = 4

#测试路径
load_time = '20210611_173541'
load_file = 'kfold_jun11_revised_crop1_725_aug_p_0_attri_7_8resnext101_32x8d_101pretrain-Falsesize224_4'

load_model_path = os.path.join(save_dir, load_time+load_file,'model_param.pkl')
anno_csv_path = "may9_936_after_revise2_crop_{}_{}train_kfold.csv".format(crop_method, dataset_size)  #may9_936_after_revise2_crop_1_{}train_kfold.csv".format(dataset_size)#1_936_nov_18_725train_output.csv"
test_anno_csv_path = "may9_936_after_revise2_crop_0_{}train_kfold.csv".format(dataset_size) #may9_936_after_revise2_crop_1_{}train_kfold.csv".format(dataset_size)


##只改这里
use_part = 1#(比如part1-1)，对应need_attributes_idx_total中的第(use_part+1)行
use_gpu = '0' #str(use_part%8) 通过nvidia-smi命令查看空闲的gpu编号，一个编号是一张卡，不要一次占两张卡
need_attributes_idx_total = [[7,8], #牙齿角度的标号\
                              [4,5,6], #选择在处理之后的表格中的列数 如角度在处理后的表格中是4,5,6列 基骨在处理后的表格中是9,30,32\ smr中是567列
                              [14,17,20,29,26,23],\
                              [15,18,21,30,27,24],\
                              [16,19,22],\
                              [31,28,25],
                              [10,11],
                             [12],
                             [13]]
save_name = 'part{}_jun11_revised_crop{}_{}_aug_p_{}_attri_{}_{}'.format(use_part,crop_method, dataset_size,flip_prob,need_attributes_idx_total[use_part][0],need_attributes_idx_total[use_part][-1])+ model_name+'_'+ model_size+"pretrain-"+str(pretrain)+"size"+str(INPUT_SIZE[0])
test_save_name = 'test'+time_first+save_name  #'test'+load_time+load_file # 'kfold_may5_revised_crop1_{}_aug_p_{}_attri_{}_{}'.format(dataset_size, flip_prob,need_attributes_idx_total[0][0],need_attributes_idx_total[0][-1])+ model_name+'_'+ model_size+"pretrain-"+str(pretrain)
file_dir_test = 'test_files\\'+test_save_name

for i in range(len(need_attributes_idx_total)):
    for j in range(len(need_attributes_idx_total[i])):
        need_attributes_idx_total[i][j] -= 0
need_attributes_idx = need_attributes_idx_total[use_part]
max_epoch = 250
use_uniform_mean = '12'

#如果不是4个牙位一起算的话，use uniform要等于use_attribute
