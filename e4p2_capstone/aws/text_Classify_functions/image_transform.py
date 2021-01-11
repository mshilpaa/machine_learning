from PIL import Image
import numpy as np
import io
from zipfile import ZipFile 
import zipfile
import os
import json
import shutil
import boto3
from utils import *


def transform_image(image_bytes):
    try:
        image = toSquare_img(image_bytes)
        # print('Successfully padded the image')
        # image = Image.open(io.BytesIO(image_bytes))
        newsize = (224,224) 
        return image.resize(newsize)
    except Exception as e:
        print(repr(e))
        raise(e)

def image_transformations(decoded,ACCESS_KEY,SECRET_KEY,BUCKET_NAME):
    try:
        print('img-transforming -------')
        username =( decoded.parts[1].content).decode('utf-8')
        print(username)
      
       
        path = f'/tmp/dataset' # in lambda
        FILE_NAME =f'/tmp/dataset.zip' 
        S3_FILE_NAME = f'{username}/img_classify/dataset.zip'
        S3_ZIP_NAME = f'{username}/img_classify/raw_dataset.zip'
        try:
            os.mkdir(path)
        except:
            shutil.rmtree(path)
            os.mkdir(path)
        file_class = decoded.parts[2].content
        file_class = json.loads(file_class)
        f = open(f'{path}/label.txt','a')
        for i in file_class:
            f.write(f'{i} {file_class[i]} \n')
        f.close()
        # print('generated txt file')
        
        session = boto3.session.Session(aws_access_key_id=ACCESS_KEY,aws_secret_access_key=SECRET_KEY)
        s3 = session.resource("s3")
        bucket = s3.Bucket(BUCKET_NAME)
        obj = bucket.Object(S3_ZIP_NAME)
        
        with io.BytesIO(obj.get()["Body"].read()) as tf:
            tf.seek(0)
            with zipfile.ZipFile(tf, mode='r') as zipf:
                for filename in zipf.namelist():
                    # print(filename.split('/')[-1])
                    if filename[-1] !='/':
                        image_bytes = zipf.read(filename)
                        image = transform_image(image_bytes)
                        image.save(f'{path}/{filename.split("/")[-1]}')
        
        
        # print('padding complete')

        

        with ZipFile(FILE_NAME,'a', compression = zipfile.ZIP_DEFLATED) as zip: 
    
            for file in os.listdir(path): 
                zip.write(f'{path}/{file}') 
       
        print('zipping complete')    

        upload_to_s3(FILE_NAME,BUCKET_NAME,S3_FILE_NAME,ACCESS_KEY,SECRET_KEY) 
        shutil.rmtree(path)
        os.remove(FILE_NAME)
        print('removed folder')
        return 1
    except Exception as e:
        print(repr(e))
        return 0
        