

from torch.utils.data import Dataset
import random
import numpy as np
import torch
from PIL import Image
from zipfile import ZipFile 


def unzip_files(filename):
  with ZipFile(filename, 'r') as zip_file:    
      # extracting all the files 
      print('Extracting all the files now...') 
      zip_file.extractall() 
      print('Done!')

def get_data(label_file,img_dir ):
    images = []
    target = []

    labels = (open(label_file,'r')).readlines()[:1000]


    for label in labels:
      a  = label.split(' ')
      images.append(f'{img_dir}/{a[0]}')
      l = a[1].split('\n')
      target.append(l[0])         
    dataset =  list(zip(images,target))
    random.shuffle(dataset)
    if len(dataset)<200:
      train_split = 80
    else:
      train_split = 70
    train_len = len(dataset)*train_split//100
    train = dataset[:train_len]
    test = dataset[train_len:]
    return train,test

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        
        self.transform = transform
        
        self.input_images,target = zip(*data) 
        self.target = np.asarray(target)
        self.target = torch.from_numpy(self.target.astype('long'))

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
  
        # print(self.target[idx],'---',idx,'----')
        input_image = np.asarray(Image.open(self.input_images[idx]))
        target = self.target[idx]
        
        if self.transform:
            input_image = self.transform(image=input_image)['image']
                    
        return input_image,target

def form_data(data_path,label_file,img_dir , train_transform,test_transform ,unzip=False):

  if unzip == True:
    unzip_files(filename=data_path)
  else:
    print('Files already downloaded')
  print('Forming the dataset')
  train, test = get_data(label_file=label_file,img_dir=img_dir)

  train_set = CustomDataset(train,transform=train_transform )
  test_set = CustomDataset(test,transform=test_transform )
  print('Done!')
  return train_set, test_set
