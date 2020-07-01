import torch 
import torchvision
import os
from torchvision import transforms
from PIL import Image
import json
from pathlib import Path
from django.conf import settings
# import albumentations as A
# import albumentations.pytorch as AP
from .model_architecture import Unet_Resnet
from torchvision.utils import save_image



def Predict(media_root=settings.MEDIA_ROOT+'images/', fgbg_file='fg-bg80001.jpg', bg_file='bg1.jpg'):

    print(fgbg_file,bg_file,'----------------------predict inference.py')
    aug = transforms.Compose([transforms.Resize((64, 64)),                                   
                                    transforms.ToTensor(),                     
                            ])

    fgbg_path = Path(media_root + '/' + fgbg_file)
    bg_path = Path(media_root + '/' + bg_file)

    fgbg = Image.open(fgbg_path)
    bg = Image.open(bg_path)   

    fgbg = aug(fgbg)
    bg = aug(bg)

    test_data = torch.cat((fgbg,bg),0)
    test_data = test_data.unsqueeze(0)
    
    print('fgbg dim',fgbg.size())
 

    model_path = os.path.join(settings.MODELS,'model_best3.pt')
    checkpoint = torch.load(model_path,map_location='cpu')
    # os.environ['TORCH_HOME'] = torch_home
    # resnet = 
    # resnet.eval()
    # output = resnet(image)
    # value, index = torch.max(output, 1)
     
    model = Unet_Resnet()
    model.load_state_dict(checkpoint['state_dict'],strict=False)
    model.eval()
    mask,depth = model(test_data)

    mask1 = mask.squeeze(0).permute(0, 2, 1).cpu()
    depth1 = depth.squeeze(0).permute(0, 2, 1).cpu()
    print(mask1.shape)
    mask_path = settings.MEDIA_ROOT+'images/'+'mask-'+fgbg_file+'.png'
    depth_path = settings.MEDIA_ROOT+'images/'+'depth-'+fgbg_file+'.png'
    save_image(mask1,mask_path )
    save_image(depth1,depth_path )
    
    return fgbg_path,bg_path,mask_path,depth_path
