# python img_classify.py userid model_name classes

# data loader


import torch
import albumentations as A
import albumentations.pytorch as AP
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn as nn
import os
import boto3
import sys
import shutil

from custom_dataset import form_data
from data_load import load
from train_test import *
from save_model import save_model_cpu
from s3_upload import upload_to_s3




USERID = sys.argv[1]
try:
  os.mkdir('tmp')
except:
  shutil.rmtree('tmp')

LABEL_FILE = f'tmp/dataset/label.txt'
IMAGE_DIR = f'tmp/dataset'
MODEL_NAME =  sys.argv[2] #'resnet18' # or 'mobilenetv2'
CLASSES =int(sys.argv[3])
MODEL_SAVE_PATH = f'tmp/{MODEL_NAME}.pt'

ACCESS_KEY = ''
SECRET_KEY = ''
BUCKET_NAME = ''
S3_ZIP_NAME = f'{USERID}/img_classify/dataset.zip'
LOCAL_ZIP_NAME = 'dataset.zip'
S3_MODEL_PATH =f'{USERID}/img_classify/{MODEL_NAME}.pt'
S3_MODEL_TXT_FILE_PATH = f'{USERID}/img_classify/{MODEL_NAME}_stats.txt'
LOCAL_MODEL_TXT_FILE_PATH = f'{MODEL_NAME}_stats.txt'
EPOCHS = 1

s3 = boto3.client('s3', aws_access_key_id= ACCESS_KEY,
                  aws_secret_access_key= SECRET_KEY)
s3.download_file(BUCKET_NAME,S3_ZIP_NAME,LOCAL_ZIP_NAME)

mean = (0.53105756 , 0.58601165 , 0.61593276)

std = (0.28278487,  0.25762487 , 0.29407342)
train_transform = A.Compose(
    [

     A.Resize(224, 224, interpolation=1, always_apply=True, p=1),
     A.Flip(always_apply=False, p=0.5),
     A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.1, rotate_limit=45,
                        interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
     A.RandomContrast(limit=0.2, always_apply=False, p=0.5),
    #  A.ChannelShuffle(always_apply=False, p=0.5),
    #  A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=1, border_mode=4,
    #                   value=None, mask_value=None, always_apply=False, p=0.5),
     A.Cutout(num_holes=4, max_h_size=32,max_w_size = 32,p=1,fill_value=0.5*255),
     A.Normalize(mean=mean, std=std),
     AP.ToTensor()
             ])

test_transform = A.Compose(
    [
     A.Resize(224, 224, interpolation=1, always_apply=True, p=1),
     A.Normalize(mean=mean, std=std),
     AP.ToTensor()
             ])
train_set,test_set = form_data(unzip=True,data_path = LOCAL_ZIP_NAME,label_file=LABEL_FILE,img_dir=IMAGE_DIR ,  train_transform=train_transform, test_transform=test_transform )




trainloader,testloader = load(train_set,test_set,batch_size=128)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

if MODEL_NAME ==  'mobilenetv2':
  print('model:mobilenetv2')
  model = torch.hub.load('pytorch/vision', 'mobilenet_v2', pretrained=True).to(device)
  model.classifier[1] = nn.Sequential(
                        nn.Linear(1280, 256),
                        nn.ReLU(),
                        nn.Linear(256, CLASSES),
                        ).to(device)
elif  MODEL_NAME ==  'resnet18':
  print('model:resnet18')
  model = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True).to(device)
  model.fc = nn.Sequential(
                      nn.Linear(512, 256),
                      nn.ReLU(),
                      nn.Linear(256, CLASSES),
                      ).to(device)



optimizer = optim.SGD(model.parameters(), lr=0.001,momentum=0.9)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5, last_epoch=-1)

for epoch in range(EPOCHS):

    train_acc,train_loss = train(model, device, trainloader, optimizer, criterion)
    scheduler.step()

test_acc,test_loss = test(model, device, criterion, testloader)
save_model_cpu(model,MODEL_SAVE_PATH,device)
uploaded = upload_to_s3(MODEL_SAVE_PATH,BUCKET_NAME,S3_MODEL_PATH,ACCESS_KEY,SECRET_KEY)
print(f'Train Acc:{train_acc}')
print(f'Test Acc: {test_acc}')

shutil.rmtree('tmp')
os.remove('dataset.zip')
f = open(LOCAL_MODEL_TXT_FILE_PATH,'a')
f.write(f"{test_acc} {train_acc} \n")
f.close()
uploading = upload_to_s3(LOCAL_MODEL_TXT_FILE_PATH, BUCKET_NAME, S3_MODEL_TXT_FILE_PATH, ACCESS_KEY, SECRET_KEY)
os.remove(LOCAL_MODEL_TXT_FILE_PATH)