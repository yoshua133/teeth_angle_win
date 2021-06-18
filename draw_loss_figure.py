import matplotlib.pyplot as plt
import csv
from IPython import embed
csv_list = list()
train_loss = list()
test_loss = list()
train_offset = []
test_offset = []
csv_path = "/home/shimr/teeth_new/crowddet_teeth/teethcode_2021_jan30/train_files/20210502_224615part0_apr30_revised_crop1_540_aug_p_0_attri_7resnext101_32x8d_101pretrain-Falsesize224/part0_apr30_revised_crop1_540_aug_p_0_attri_7resnext101_32x8d_101pretrain-Falsesize224_loss.csv"
num = 0 
with open(csv_path) as f:   
      for row in f:
          num +=1
          if num<3:
            continue   
          csv_list.append(row)
          train_loss.append(row[1])
          test_loss.append(row[3])
          train_offset.append(row[2])
          test_offset.append(row[4])

x1 = range(0, 5*len(train_loss),5)
x2 = range(0, 5*len(train_loss),5)
plt.subplot(2, 1, 1)
# plt.plot(x1, y1, 'o-',color='r')
plt.plot(x1, train_offset, 'o-',label="Train_Accuracy")
plt.plot(x1, test_offset, 'o-',label="Valid_Accuracy")
plt.title('Test accuracy vs. epoches')
plt.ylabel('Test accuracy')
plt.legend(loc='best')
plt.subplot(2, 1, 2)
plt.plot(x2, train_loss, '.-',label="Train_Loss")
plt.plot(x2, test_loss, '.-',label="Valid_Loss")
plt.xlabel('Test loss vs. epoches')
plt.ylabel('Test loss')
plt.legend(loc='best')
#embed()
plt.savefig('./train_loss.tif')
      
        