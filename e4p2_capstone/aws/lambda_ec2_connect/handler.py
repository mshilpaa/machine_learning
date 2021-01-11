try:
    import unzip_requirements
except ImportError:
    pass

import json
import paramiko
import boto3

region = ''
PEM_FILE_PATH = ""
instances = ['i-k']

def text_classify_train(decoded,instances,PEM_FILE_PATH):

    host = decoded['host']
    userid = decoded['userid']
    num_classes = decoded['num_classes']
    command = f"python3 text_classify.py {userid} {num_classes}"
    k = paramiko.RSAKey.from_private_key_file(PEM_FILE_PATH )
    # print('5')
    c = paramiko.SSHClient()
    # print('6')
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # print('7')
    # print("Connecting to " + host)
    c.connect( hostname = host, username = "ubuntu", pkey = k )
    print("Connected to " + host)

    # for command in commands:
    # print(command)
    stdin , stdout, stderr = c.exec_command(command)
    
    return stdout, stderr

def img_classify_train(decoded,instances,PEM_FILE_PATH):

    userid = decoded['userid']
    num_classes = decoded['num_classes']
    model_name = decoded['model_name']
    host = decoded['host']
    command = f"python3 image_classify.py {userid} {model_name} {num_classes}"
    print('command',command)
    # print('3')
    # s3_client.download_file('Add your bucket name','key/Add your key name.pem', '/tmp/Add your key name.pem')
    # print('4')
    k = paramiko.RSAKey.from_private_key_file(PEM_FILE_PATH )
    # print('5')
    c = paramiko.SSHClient()
    # print('6')
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # print('7')
    # print("Connecting to " + host)
    c.connect( hostname = host, username = "ubuntu", pkey = k )
    print("Connected to " + host)

    # for command in commands:
    # print(command)
    stdin , stdout, stderr = c.exec_command(command)
    
    return stdout, stderr

def start_ec2(decoded,region,instances):
    try:
        print('starting ec2 func')
        # cmd = []
        # for i in range (1,len(decoded.parts)):
        
        # cmd = json.loads(event['body'])['data']
        # print('cmd',cmd)
        ec2_instance = boto3.client('ec2',region_name=region)
        # print('s')
        ec2_instance.start_instances(InstanceIds=instances)
        # print('started instance')
        host = 0
        ec2 = boto3.resource('ec2')
        running_instances = ec2.instances.filter(InstanceIds=instances )
        for instance in running_instances:
            host = instance.public_ip_address
        return host
    except Exception as e:
        print(repr(e))

def stop_ec2(instances,region):
    try:
        ec2_instance = boto3.client('ec2',region_name=region)
        ec2_instance.stop_instances(InstanceIds=instances)
        return 1
    except Exception as e:
        print(repr(e))

def lambda_ec2_connect(event,context):
    try:

        print(event)
        decoded = json.loads(event['body'])
        print(decoded)
        function_usage = decoded['function_usage']
        print(function_usage)
        if function_usage == 'start_ec2':
            host = start_ec2(decoded,region,instances)
        
            return {
                    'statusCode': 200,
                    'headers':{
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Credentials': True
                    },
                    'body': json.dumps({'Status':'1','output':str(host) })
                }
        elif function_usage == 'img_classify_train':
            stdout,stderr = img_classify_train(decoded,instances,PEM_FILE_PATH)
            a = stdout.read()
            b = stderr.read()
            print(a)
            print(b)
            return {
                    'statusCode': 200,
                    'headers':{
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Credentials': True
                    },
                    'body': json.dumps({'Status':'1','output':a.decode('utf-8'),'error':b.decode('utf-8') })
                }
        elif function_usage == 'text_classify_train':
            stdout,stderr = text_classify_train(decoded,instances,PEM_FILE_PATH)
            a = stdout.read()
            b = stderr.read()
            return {
                    'statusCode': 200,
                    'headers':{
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Credentials': True
                    },
                    'body': json.dumps({'Status':'1','output':a.decode('utf-8'),'error':b.decode('utf-8') })
                }
        elif function_usage == 'stop_ec2':
            res = stop_ec2(instances,region)
            return {
                    'statusCode': 200,
                    'headers':{
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*',
                        'Access-Control-Allow-Credentials': True
                    },
                    'body': json.dumps({'Status':'1','output':str(res) })
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
