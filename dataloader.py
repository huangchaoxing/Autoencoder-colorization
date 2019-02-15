# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:51:04 2019

@author: HP
"""
import numpy as np
import torchvision
import random
from torchvision import datasets
from torchvision import transforms
import torch
from torch.utils.data import random_split
from torch.utils.data import Dataset
from scanfile import ScanFile
from PIL import Image
from skimage.color import rgb2lab,rgb2gray,lab2rgb
data_dir="testSetPlaces205_resize/testSet_resize"
dir_list=ScanFile(data_dir).scan_files()
dir_list=sorted(dir_list,key=lambda k:random.random())

class scenedataset(Dataset):
    def __init__(self,dir_list,flag):
        
        num=len(dir_list)
        #normalization=transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
        #self.feature_trans=transforms.Compose([transforms.Resize([299,299]),
#                                               transforms.Grayscale(num_output_channels=1),
 #                                              transforms.ToTensor(),normalization])
        if flag=="train":
            self.sub_dir_list=dir_list[:int(0.8*num)]
            print("train number:",len(self.sub_dir_list))
            self.trans=transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),transforms.RandomResizedCrop(224)])
        if flag=="validation":
            self.sub_dir_list=dir_list[int(0.8*num):]
            print("validation number:",len(self.sub_dir_list))
            self.trans=transforms.Compose([transforms.RandomHorizontalFlip(p=0),transforms.Resize([224,224])])
        if flag=="test":
            self.sub_dir_list=dir_list[0:num]
            print("test number:",len(self.sub_dir_list))
            self.trans=transforms.Compose([transforms.Resize([224,224])])
        if flag=="pre-process":
            self.sub_dir_list=dir_list
            print("pre-process/test number:",len(self.sub_dir_list))
            self.trans=transforms.Compose([transforms.RandomHorizontalFlip(p=0),transforms.Resize([299,299])])
    def __getitem__(self,idx):
        image_path=self.sub_dir_list[idx]
        image_raw=Image.open(image_path)
        
       
        #image_ab=torch.from_numpy(image_ab.transpose((2,0,1)))
        image_original=self.trans(image_raw)
        image_original=np.asarray(image_original)
        image_lab=(rgb2lab(image_original)+128)/255
        image_ab=image_lab[:,:,1:3]
        image_ab=torch.from_numpy(image_ab.transpose((2,0,1))).float()
        
        #image_original=self.data_trans(image_original)
        image_original=rgb2gray(image_original)
        image_original=torch.from_numpy(image_original).unsqueeze(0)        
        data=image_original
        label=image_ab
        #print(label.max())
        
       # print(data[0].shape)
        return data,label
    
    def __len__(self):
        return len(self.sub_dir_list)
if  __name__=="__main__":    
    training_set=scenedataset(dir_list,flag="train")    
    training_loader=torch.utils.data.DataLoader(training_set,batch_size=2,num_workers=0)
    #print(len(training_loader.dataset))
    
    import matplotlib.pyplot as plt
    def reconstruct_rgb(grey_scale,ab_img):
         
         color_image = torch.cat((grey_scale, ab_img), 0).numpy() # combine channels
         color_image = color_image.transpose((1, 2, 0))  # rescale for matplotlib
         color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
         color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128   
         color_image = lab2rgb(color_image.astype(np.float64))
         print(grey_scale.max())
         print(grey_scale.min())
         grey_scale = grey_scale.squeeze().numpy()
         plt.figure(1)
         plt.imshow(grey_scale,cmap='gray')
         plt.figure(2)
         plt.imshow(color_image)
         
 
      
    # get some random training images
    dataiter = iter(training_loader)
    show,label=dataiter.next()
    print("haha")
    print("image number:",len(show))
    print("label number",len(label))
#    print(show.max())
#    print(show.min())
    grey_input=show[0]
    ab_img=label[0]
    reconstruct_rgb(grey_input.double(),ab_img.double())