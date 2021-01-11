
import boto3
from botocore.exceptions import NoCredentialsError

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
