try:
    import unzip_requirements
    print('unziped requirements')
except ImportError:
    pass

import boto3
import base64
import json
from requests_toolbelt.multipart import decoder
import os
from image_transform import image_transformations
from upload_csv import *


S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else ''
ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_KEY = os.environ['SECRET_KEY']

def text_classify_functions(event,context):
    try:
        print('event-------',event)

        content_type_header = event['headers']['content-type']
        
        body = base64.b64decode(event['body'])
        print('body-------',body)
        print('body loaded')
        print(content_type_header)
        
        decoded = decoder.MultipartDecoder(body,content_type_header)
        function_usage =( decoded.parts[0].content).decode('utf-8')
        if function_usage == 'for_csv_upload':
            res = UploadCsv(event,context,ACCESS_KEY,SECRET_KEY,S3_BUCKET)
            print(res)
            return {
            'statusCode': 200,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'status': '1', 'result': str(res) })
        }
        elif function_usage == 'for_images_upload':
            res = image_transformations(decoded,ACCESS_KEY,SECRET_KEY,S3_BUCKET )
            return {
            'statusCode': 200,
            'headers':{
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'Status': str(res)  })
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
            'body': json.dumps({ 'error': repr(e) })
        }
