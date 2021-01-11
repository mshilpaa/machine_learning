# import spacy
# nlp = spacy.load('en')
import torch
from torchtext import vocab
import pickle
VOCAB_PATH = '/content/tokenizer.pickle'
vocab_file = open(VOCAB_PATH, 'rb')      
vocabs = pickle.load(vocab_file) 
model1 = torch.jit.load('/content/saved_weights.pt')

def predict_sentiment(model,sentence,min_len=5):
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

sentence = 'hello very good morning to all'
prediction = predict_sentiment(model1,sentence)