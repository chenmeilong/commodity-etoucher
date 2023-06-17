import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
import numpy as np
import cv2

##数据增强
transform = transforms.Compose([

    transforms.ToPILImage(),
    transforms.RandomGrayscale(p=0.2),  # 随机变为灰度图像
    transforms.ColorJitter(brightness=0.3),  # ，1 表示原图  随机亮度
    transforms.ColorJitter(contrast=0.3),  # 随机从 0 ~ 2 之间对比度变化，1 表示原图
    transforms.ColorJitter(hue=0.5),# 随机从 -0.5 ~ 0.5 之间对颜色变化
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])   #归一化

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf

class BagDataset(Dataset):

    def __init__(self, transform=None):
        self.transform = transform
        
    def __len__(self):
        return len(os.listdir('data/train_data'))

    def __getitem__(self, idx):
        img_name = os.listdir('data/train_data')[idx]
        imgA = cv2.imread('data/train_data/'+img_name)
        imgA = cv2.resize(imgA, (480, 480))
        imgB = cv2.imread('data/label_data/'+img_name, 0)
        imgB = cv2.resize(imgB, (480, 480))
        imgB = imgB/255
        imgB = imgB.astype('uint8')
        imgB = onehot(imgB, 2)
        imgB = imgB.transpose(2,0,1)
        imgB = torch.FloatTensor(imgB)
        #print(imgB.shape)
        if self.transform:
            imgA = self.transform(imgA)
        return imgA, imgB

bag = BagDataset(transform)

train_size = int(0.9 * len(bag))
test_size = len(bag) - train_size
train_dataset, test_dataset = random_split(bag, [train_size, test_size])

train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=2)


if __name__ =='__main__':

    for train_batch in train_dataloader:
        print(train_batch)

    for test_batch in test_dataloader:
        print(test_batch)
