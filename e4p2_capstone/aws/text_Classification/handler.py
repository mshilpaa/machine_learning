try:
    import unzip_requirements
except ImportError:
    pass


import base64
import json
from requests_toolbelt.multipart import decoder
from text_classify import *
import os

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else ''
ACCESS_KEY = os.environ['ACCESS_KEY']
SECRET_KEY = os.environ['SECRET_KEY']




def text_classification(event,context):
    try:
        print('event-------',event)

        content_type_header = event['headers']['content-type']
        
        body = base64.b64decode(event['body'])
        print('body-------',body)
        print('body loaded')
        print(content_type_header)
        
        decoded = decoder.MultipartDecoder(body,content_type_header)
        function_usage =( decoded.parts[0].content).decode('utf-8')
        if function_usage == 'text_classify_test':
            prediction = predict_class(S3_BUCKET,decoded,ACCESS_KEY,SECRET_KEY,min_len=5)
            return {
                'statusCode': 200,
                'headers':{
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                    'Access-Control-Allow-Credentials': True
                },
                'body': json.dumps({'status': '1', 'result': str(prediction) })
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