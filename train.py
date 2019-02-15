# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 11:47:10 2019

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
from dataloader import scenedataset
from model import  ColorizationNet
from scanfile import ScanFile
import time
from tensorboardX import SummaryWriter
writer = SummaryWriter('log1')
weight_test=[]
LEAD=55
LR=9e-4
BATCH_SIZE=64
ITERATE_NUM=46
data_dir="testSetPlaces205_resize/testSet_resize"
#data_dir="dummy_data"
dir_list=ScanFile(data_dir).scan_files()

#dir_array=np.array(dir_list)


dir_list=sorted(dir_list,key=lambda k:random.random())

training_set=scenedataset(dir_list,flag="train")    
training_loader=torch.utils.data.DataLoader(training_set,batch_size=BATCH_SIZE,num_workers=0)

validation_set=scenedataset(dir_list,flag="validation")
validation_loader=torch.utils.data.DataLoader(validation_set,batch_size=BATCH_SIZE,num_workers=0)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")
print(device)

#model=ColorizationNet()
model=torch.load("54_net.pt")
model=model.to(device)
print("model ready")

criterion=nn.MSELoss()
optimizer=optim.Adam(filter(lambda p: p.requires_grad,model.parameters()),lr=LR,betas=(0.9,0.999))
step=28421
start_time=time.time()

train_loss_result={}
epoch_valid_loss_result={}

for epoch in range(ITERATE_NUM):
    epoch_loss=0
    epoch_valid_loss=0
    running_loss=0.0
    print("This is epoch",epoch+1)
    batch_num=0
    step_loss=0
    for data in training_loader:
        
        inputs, labels = data
        labels = labels.to(device,dtype=torch.float)
        optimizer.zero_grad()
        
        image=inputs.to(device,dtype=torch.float)
        
#        print(type(features),type(image))
#        while(1):
#            pass
        outputs=model(image).to(device,dtype=torch.float)
        loss=criterion(outputs,labels)
        
        loss.backward()
        optimizer.step()
        #print(loss.item())
        epoch_loss+=loss.item()
        step_loss+=loss.item()
        batch_num=batch_num+1
        
        running_loss = 0.0
        step=step+1
        
        if step%20==0:
            print("loss:",step_loss/20,"step:",step)
            writer.add_scalar('training_loss vs steps',step_loss/20,step)
            step_loss=0
        #print("avg duration is",duration/batch_num) 
        #print("epoch time is",time.time()-epoch_start_time)
    epoch_loss=epoch_loss/(31943/BATCH_SIZE)
    
        #train_loss_result[epoch]=epoch_loss
    train_loss_result[epoch]=epoch_loss    
    if epoch%3 ==0:    
      torch.save(model,str(epoch+LEAD)+'_net.pt')  
      with torch.no_grad():
         for i,data in enumerate(validation_loader):
             inputs,label=data
             label = label.to(device,dtype=torch.float)
             image=inputs.to(device,dtype=torch.float)
             outputs=model(image).to(device,dtype=torch.float)
             valid_loss=criterion(outputs,label)
             epoch_valid_loss += valid_loss.item()  
      epoch_valid_loss=epoch_valid_loss/(7986/BATCH_SIZE) 
      epoch_valid_loss_result[epoch]=epoch_valid_loss  
     
      writer.add_scalars('validation LOSS V.S epochs',{'validation':epoch_valid_loss,'training':epoch_loss},epoch+49)
    writer.add_scalar('train loss V.S EPOCHS',epoch_loss,
                                          epoch+49)    
    print("*****************************************")    
    print ("The epoch loss is",epoch_loss,"EPOCH",epoch+50) 
    #print("e",model.encode.conv1.weight.requires_grad)
    #print(model.encode.conv1.weight)
    #print("fe",model.extract[14].conv.weight.requires_grad)
    print("***************************************")

train_epoch_value=list(train_loss_result.keys())
validation_epoch_value=list(epoch_valid_loss_result.keys())

train_loss_value=list(train_loss_result.values())
valid_loss_value=list(epoch_valid_loss_result.values())

plt.figure(1)
plt.plot(train_epoch_value,train_loss_value,)
plt.show()
plt.figure(2)
plt.plot(validation_epoch_value,valid_loss_value)
plt.show()
       