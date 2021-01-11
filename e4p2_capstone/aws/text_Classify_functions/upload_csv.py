import base64
from requests_toolbelt.multipart import decoder
import os
import pandas as pd 
from utils import *


def UploadCsv(event,context,ACCESS_KEY,SECRET_KEY,BUCKET_NAME):
    try:
        content_type_header = event['headers']['content-type']
        
        body = base64.b64decode(event['body'])
        username = decoder.MultipartDecoder(body,content_type_header).parts[1].content.decode('utf-8')
        print('username----',username)
        csv_file_name = 'dataset.csv'
        csv_file = decoder.MultipartDecoder(body,content_type_header).parts[2].content.decode('utf-8')
        csv_file_path = f'/tmp/{csv_file_name}'
        print("csv_File------ ",csv_file)
        f = open(csv_file_path,"w")
        f.write(csv_file)
        f.close()
        print('created csv file')
        df = pd.read_csv(csv_file_path,names=['data','categories'])
        df.fillna('No value',inplace=True)
        codes,index = pd.factorize(df['categories'])
        df['labels']= codes
        df = df.drop(['categories'],axis=1)
        df.to_csv(csv_file_path,index=False,header=False)
        print('created clean csv')
        S3_FILE_NAME = f'{username}/{csv_file_name}'

        upload_to_s3(csv_file_path ,BUCKET_NAME,S3_FILE_NAME,ACCESS_KEY,SECRET_KEY) 
        print('uploaded to s3')
        os.remove(csv_file_path)
        print('removed file')
        return tuple(index)
    except Exception as e:
        print(repr(e))
        #return repr(e)