import torch
import torch.nn as nn
import torch.nn.functional as F

class Resnet_Block(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Resnet_Block, self).__init__()
        # Input Block

        self.conv1 = nn.Sequential(
          
          nn.BatchNorm2d(in_channels),
          nn.ReLU(),
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False), 
          
          nn.BatchNorm2d(out_channels),
          nn.ReLU(),
          nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False), 
      )
        
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False),
        )

    def forward(self,x):
      out = self.conv1(x)
      
      one = self.shortcut(x)
      # print(out.size(),x.size(),one.size())
      out += one
      return out

class Downsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Downsample, self).__init__()
        # Input Block
        
        self.conv1 = nn.Sequential(
          
          # nn.BatchNorm2d(in_channels),
          # nn.ReLU(),
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False,stride=2), 
          
          # nn.BatchNorm2d(out_channels),
          # nn.ReLU(),
          # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False), 
      )
        
        # self.shortcut = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False,stride=2),
        # )

    def forward(self,x):
      out = self.conv1(x)
      
      # one = self.shortcut(x)
      # print('---',out.size(),x.size(),one.size())
      # out += one
      return out

class Upsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Upsample, self).__init__()
        # Input Block
        
        self.conv1 = nn.Sequential(
          
          # nn.BatchNorm2d(in_channels),
          # nn.ReLU(),
          nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2, 2), padding=0, bias=False,stride=2), 
          
          # nn.BatchNorm2d(out_channels),
          # nn.ReLU(),
          # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False), 
      )
        
        # self.shortcut = nn.Sequential(
        #     nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1), padding=0, bias=False,stride=2),
        # )

    def forward(self,x):
      out = self.conv1(x)
      
      # one = self.shortcut(x)
      # print('---',out.size(),x.size(),one.size())
      # out += one
      return out

def final_layer(in_channels,out_channels):
  return nn.Sequential(
          
          nn.BatchNorm2d(in_channels),
          nn.ReLU(),
          nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False), 
          nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1, bias=False), 
  )  
  
class Unet_Resnet(nn.Module):
    def __init__(self):
        super(Unet_Resnet, self).__init__()
        # Input Block

        self.conv1 = nn.Conv2d(in_channels=6, out_channels=64, kernel_size=(3, 3), padding=1, bias=False)
      
       
        
        self.enc1 = Resnet_Block(64,64)
        self.enc2 = Resnet_Block(128,128)
        self.enc3 = Resnet_Block(128,128)
        self.enc4 = Resnet_Block(256,256)
        self.enc5 = Resnet_Block(256,256)

        self.enc1_down = Downsample(64,128)       
        self.enc2_down = Downsample(128,128)
        self.enc3_down = Downsample(128,256)
        self.enc4_down = Downsample(256,256)

        self.dec1_1 = final_layer(64,1)
        self.dec1 = Resnet_Block(128,64)
        self.dec2 = Resnet_Block(256,128)
        self.dec3 = Resnet_Block(256,128)
        self.dec4 = Resnet_Block(512,256)
        # self.dec5 = Resnet_Block(128,128)


        self.dec1_up = Upsample(128,64)       
        self.dec2_up = Upsample(128,128)
        self.dec3_up = Upsample(256,128)
        self.dec4_up = Upsample(256,256)


        self.Mdec1_1 = final_layer(64,1)
        self.Mdec1 = Resnet_Block(128,64)
        self.Mdec2 = Resnet_Block(256,128)
        self.Mdec3 = Resnet_Block(256,128)
        self.Mdec4 = Resnet_Block(512,256)
        # self.Mdec5 = Resnet_Block(128,128)


        self.Mdec1_up = Upsample(128,64)       
        self.Mdec2_up = Upsample(128,128)
        self.Mdec3_up = Upsample(256,128)
        self.Mdec4_up = Upsample(256,256)

        self.sigmoid = nn.LogSigmoid()
        self.relu = nn.ReLU()

    def forward(self,x):

      out1 = self.conv1(x)       # o/p 64x64x64
      # print('out1',out1.size())
      en1 = self.enc1(out1)       # 64x64x64
      # print('en1',en1.size())
      en1_1 = self.enc1_down(en1)   # 128x32x32
      # print('en1_1',en1_1.size())
      en2 = self.enc2(en1_1)       # 128x32x32
      # print('en2',en2.size())
      en2_1 = self.enc2_down(en2)  # 128x16x16
      # print('en2_1',en2_1.size())
      en3 = self.enc3(en2_1)       # 128x16x16
      # print('en3',en3.size())
      en3_1 = self.enc3_down(en3)  # 256x8x8
      # print('en3_1',en3_1.size())

      en4 = self.enc4(en3_1)       # 256x8x8
      # print('en4',en4.size())
      en4_1 = self.enc4_down(en4)  # 256x4x4
      # print('en4_1',en4_1.size())

      en5 = self.enc5(en4_1)       # 512x4x4
      # print('en5',en5.size())

# depth
      de4 = self.dec4_up(en5)     # 128x8x8
      # print('de4',de4.size())
      de4_1 = torch.cat((en4,de4 ),1)             # 256x8x8
      # print('de4_1',de4_1.size())

      de4_1 = self.dec4(de4_1)  # 128x8x8
      # print('de4_1',de4_1.size())

      de3 = self.dec3_up(de4_1)  # 64x16x16
      # print('de3',de3.size())
      de3_1 = torch.cat((en3,de3 ),1)                 # 128x16x16
      # print('de3_1',de3_1.size())

      de3_1 = self.dec3(de3_1)   # 64x16x16
      # print('de3_1',de3_1.size())
      
      de2 = self.dec2_up(de3_1)  # 64x32x32
      # print('de2',de2.size())

      de2_1 = torch.cat((en2,de2 ),1)         # 128x32x32
      # print('de2_1',de2_1.size())

      de2_1 = self.dec2(de2_1)  # 64x32x32
      # print('de2_1',de2_1.size())
      
      de1 = self.dec1_up(de2_1) # 32x64x64
      # print('de1',de1.size())
      de1_1 = torch.cat((en1,de1),1) # 64x64x64
      # print('de1_1',de1_1.size())

      de1_1 = self.dec1(de1_1) # 32x64x64
      # print('de1_1',de1_1.size())
      de1_2 = self.dec1_1(de1_1) # 3x64x64
      # print('de1_2',de1_2.size())
    

# mask

      Mde4 = self.Mdec4_up(en5)     # 128x8x8
      # print('Mde4',Mde4.size())
      Mde4_1 = torch.cat((en4,Mde4 ),1)             # 256x8x8
      # print('Mde4_1',Mde4_1.size())

      Mde4_1 = self.Mdec4(Mde4_1)  # 128x8x8
      # print('Mde4_1',Mde4_1.size())

      Mde3 = self.Mdec3_up(Mde4_1)  # 64x16x16
      # print('Mde3',Mde3.size())
      Mde3_1 = torch.cat((en3,Mde3 ),1)                 # 128x16x16
      # print('Mde3_1',Mde3_1.size())

      Mde3_1 = self.Mdec3(Mde3_1)   # 64x16x16
      # print('Mde3_1',Mde3_1.size())
      
      Mde2 = self.Mdec2_up(Mde3_1)  # 64x32x32
      # print('Mde2',Mde2.size())

      Mde2_1 = torch.cat((en2,Mde2 ),1)         # 128x32x32
      # print('Mde2_1',Mde2_1.size())

      Mde2_1 = self.Mdec2(Mde2_1)  # 64x32x32
      # print('Mde2_1',Mde2_1.size())
      
      Mde1 = self.Mdec1_up(Mde2_1) # 32x64x64
      # print('Mde1',Mde1.size())
      Mde1_1 = torch.cat((en1,Mde1),1) # 64x64x64
      # print('Mde1_1',Mde1_1.size())

      Mde1_1 = self.Mdec1(Mde1_1) # 32x64x64
      # print('Mde1_1',Mde1_1.size())
      Mde1_2 = self.Mdec1_1(Mde1_1) # 3x64x64
      # print('Mde1_2',Mde1_2.size())


      

      return Mde1_2, de1_2  # mask,depth