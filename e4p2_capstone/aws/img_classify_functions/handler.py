#  ec2-65-0-248-244.ap-south-1.compute.amazonaws.com
# 65.0.248.244
# sudo apt install python3-pip python3-dev python3-venv build-essential libssl-dev libffi-dev python3-setuptools
# python3 -m venv venv
# source venv/bin/activate
# pip3 install boto3 pandas torch torchtext spacy
# python3 -m spacy download en

try:
    import unzip_requirements
except ImportError:
    pass

import boto3
import json
from requests_toolbelt.multipart import decoder
import base64
import os
from image_transform import image_transformations
from inference import classify_image

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'eva4p2-capstone'

def image_classify_functions(event,context):
    try:
        # print(event)
        content_type_header = event['headers']['content-type']
        # print(event['body'])
        body = base64.b64decode(event['body'])
        # print('body loaded')
        decoded = decoder.MultipartDecoder(body,content_type_header)
        # print(decoded)
        function_usage = decoded.parts[0].content.decode('utf-8')
        # print(function_usage)
        if function_usage == 'for_img_classify_test':
            prediction = classify_image(decoded,S3_BUCKET)
            return {
                'statusCode': 200,
                'headers':{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'Status': '1', 'Predicted Class': prediction })
        }

    except Exception as e:
        print(repr(e))
        return {
            'statusCode': 500,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({ 'Status':'0','error': repr(e)  })
        }
