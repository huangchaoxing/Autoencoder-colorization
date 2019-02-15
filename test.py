# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 11:25:18 2019

@author: HP
"""

from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
import numpy as np
import random
#from dataloader import scenedataset
from model import  ColorizationNet
from scanfile import ScanFile
from skimage.color import rgb2lab,rgb2gray,lab2rgb
from test_load import scenedataset

def reconstruct_rgb(grey_scale,ab_img,test_rank):
         
         color_image = torch.cat((grey_scale, ab_img), 0).numpy() # combine channels
         color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
         color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
         color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
         color_image = lab2rgb(color_image.astype(np.float64))
         
         grey_scale = grey_scale.squeeze().numpy()
         output_dir='testSetPlaces205_resize/his_output/'
         plt.imsave(output_dir+str(test_rank)+'_gray.jpg',grey_scale,cmap='gray')
         plt.imsave(output_dir+str(test_rank)+'_output.jpg',color_image)
        


model=torch.load("97_net.pt")

data_dir="testSetPlaces205_resize/historical"
dir_list=ScanFile(data_dir).scan_files()
test_set=scenedataset(dir_list,flag="test")    
test_loader=torch.utils.data.DataLoader(test_set,batch_size=1,num_workers=0)

#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device=torch.device("cpu")
print(device)
model=model.to(device)
test_rank=1
for data in test_loader:
    inputs= data
    #labels = labels.to(device,dtype=torch.float)
    
    image=inputs.to(device,dtype=torch.float)
    outputs=model(image).to(device,dtype=torch.float)
    
    output_ab=outputs.squeeze(0).detach()
    grey=inputs.squeeze(0).detach()
    reconstruct_rgb(grey.double(),output_ab.double(),test_rank)
    test_rank+=1
    
    