
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import boto3
import os
import io
import json
from utils import toSquare_img

print('Import End...')





def load_model(S3_BUCKET,MODEL_PATH):
    print('Downloading model...')
    s3 = boto3.client('s3')

    try:
        if os.path.isfile(MODEL_PATH)!=True:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
            print('Creating Bytestream',obj)
            bytestream = io.BytesIO(obj['Body'].read())
            print('Loading model',bytestream)
            model = torch.jit.load(bytestream)
            print('Model loaded')
            return model
    except Exception as e:
        print(repr(e))
        raise(e)


def transform_image(image_bytes):
    try:
        transformations = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.531,0.586,0.615],std=[0.282,0.257,0.294]),
            
            
        ])
        image = toSquare_img(image_bytes)
        print('Successfully padded the image')
        # image = Image.open(io.BytesIO(image_bytes))
        return transformations(image).unsqueeze(0)
    except Exception as e:
        print(repr(e))
        raise(e)

def get_prediction(image_bytes,model):
    print('applying augmentation')
    tensor = transform_image(image_bytes = image_bytes)
    return model(tensor).argmax().item()

def classify_image(decoded,S3_BUCKET):
    try:

        print('classification start')
        userid = decoded.parts[1].content.decode('utf-8')
        model_name = decoded.parts[2].content.decode('utf-8')
        MODEL_PATH = f'{userid}/img_classify/{model_name}.pt'
        model = load_model(S3_BUCKET,MODEL_PATH)
        picture = decoded.parts[3]
        prediction = get_prediction(image_bytes= picture.content,model=model)
        print('model predictions completed')
        # classes = ['Winged_Drones', 'Small_QuadCopters', 'Large_QuadCopters', 'Flying_Birds' ]
        # predicted_class = classes[int(prediction)]
        print(prediction)
        
        return prediction
    except Exception as e:
        print(repr(e))
        

