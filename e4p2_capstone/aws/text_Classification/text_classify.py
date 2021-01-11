# import spacy
# nlp = spacy.load('en')
import torch
from torchtext import vocab
import pickle
import io
import boto3
import os

def load_model(S3_BUCKET,MODEL_PATH,access_key,secret_key):
    print('Downloading model...')
    s3 = boto3.client('s3',aws_access_key_id=access_key,aws_secret_access_key=secret_key)

    try:
        if os.path.isfile(MODEL_PATH)!=True:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
            print('Creating Bytestream for model')
            bytestream = io.BytesIO(obj['Body'].read())
            print('Loading model',bytestream)
            model = torch.jit.load(bytestream)
            print('Model loaded')
            return model
    except Exception as e:
        print(repr(e))
        raise(e)

def load_vocabs(S3_BUCKET,VOCAB_PATH,access_key,secret_key):
    s3 = boto3.client('s3',aws_access_key_id=access_key,aws_secret_access_key=secret_key)

    try:
        if os.path.isfile(VOCAB_PATH)!=True:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=VOCAB_PATH)
            print('Creating Bytestream for vocab')
            bytestream = io.BytesIO(obj['Body'].read())
            vocabs = pickle.load(bytestream) 
            print('vocabs loaded')
            return vocabs
    except Exception as e:
        print(repr(e))
        raise(e)


def predict_class(S3_BUCKET,decoded,access_key,secret_key,min_len=5):
    USERID = decoded.parts[1].content.decode('utf-8')
    sentence = decoded.parts[2].content.decode('utf-8')
    
    MODEL_PATH = f'{USERID}/text_classify/saved_weights.pt'
    VOCAB_PATH = f'{USERID}/text_classify/tokenizer.pickle'   
    print(USERID,sentence,MODEL_PATH,VOCAB_PATH)
    vocabs = load_vocabs(S3_BUCKET,VOCAB_PATH,access_key,secret_key)
    model = load_model(S3_BUCKET,MODEL_PATH,access_key,secret_key)
    model.eval()
    # tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    tokenized = sentence.split()
    lent = len(tokenized)
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [vocabs[t] for t in tokenized]
    tensor1 = torch.LongTensor(indexed)
    tensor1 = tensor1.unsqueeze(0)
    token_length = torch.LongTensor([lent])
    preds = model(tensor1,token_length)
    _, prediction = torch.max(preds, 1)
    print('prediction',preds,prediction)
    return prediction.item()
