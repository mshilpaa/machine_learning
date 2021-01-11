import boto3
from botocore.exceptions import NoCredentialsError
import io
import numpy as np
from PIL import Image

def upload_to_s3(local_file,bucket,s3_file,access_key,secret_key):
  s3 = boto3.client('s3',aws_access_key_id=access_key,aws_secret_access_key=secret_key)
  try:
    s3.upload_file(local_file,bucket,s3_file)
    print("Upload Successful")
    return
  except FileNotFoundError:
    print('File not found')
    return
  except NoCredentialsError:
    print("credential not found")
    return

def toSquare_img(img_bytes):
    # print('padding the image')
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
  
    h,w = img.size[0],img.size[1]
    max_len = max(h,w)
    if h == w:
        return img
        
    elif h>w:
        diff = int(abs(h-w)/2)
        black = np.zeros((max_len,max_len))
        black_img = Image.fromarray(black,mode='RGB')

        black_img.paste(img,(0,diff))
        return black_img
    elif w>h:
        diff = int(abs(h-w)/2)
        black = np.zeros((max_len,max_len))
        black_img = Image.fromarray(black,mode='RGB')

        black_img.paste(img,(diff,0))

        return black_img