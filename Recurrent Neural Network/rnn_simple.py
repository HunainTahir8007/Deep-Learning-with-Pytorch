import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset , DataLoader
from  torch import nn
from google.colab import files
file_uploaded = files.upload()
df = pd.read_csv("/content/100_Unique_QA_Dataset.csv")
df.head()
def tokenize(text):
  text = text.lower()
  text= text.replace("?","")
  text = text.replace("'","")
  return text.split()
tokenize('Which country is known for the Eiffel Tower?')
# vocab
vocab = {"<UNK>":0}
def build_vocab(row):
  token_question = tokenize(row['question'])
  token_answer = tokenize(row['answer'])
  merge_token = token_question + token_answer
  for token in merge_token:
    if token not in vocab:
      vocab[token] = len(vocab)
df.apply(build_vocab, axis= 1)
vocab
def text_to_index(text , vocab):
  index_text = []
  for token in tokenize(text):
    if token in vocab:
      index_text.append(vocab[token])
    else:
      index_text.append(vocab["<UNK>"])
  return index_text

text_to_index("what is hunain ", {"<UNK>": 0, **{k:v for k,v in vocab.items() if k != '<UNK>'}})
class QAdataset(Dataset):
  def __init__(self, df, vocab):
    self.df = df
    self.vocab = vocab
  def __len__(self):
    return len(self.df)
  def __getitem__(self, idx):
    question =text_to_index(self.df.iloc[idx]['question'], self.vocab)
    answer = text_to_index(self.df.iloc[idx]['answer'], self.vocab)
    return torch.tensor(question) , torch.tensor(answer)
dataset = QAdataset(df , vocab)
dataload = DataLoader(dataset , batch_size=1 , shuffle=True)
class SimpleRNN(nn.Module):
    def __init__(self , vocab):
        super().__init__()
        self.embeed = nn.Embedding(vocab , 50)
        self.rnn = nn.RNN(50 , 64 , batch_first=True )
        self.lr = nn.Linear(64 , vocab)
    def forward(self , question):
        embeed = self.embeed(question)
        hidden , final = self.rnn(embeed)
        return self.lr(final.squeeze(0))
model = SimpleRNN(len(vocab))
loss_fn  = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr=0.001)
for epoch in range(20):
  train_loss = 0
  model.train()
  for qu , ans in dataload:
    y_pred = model(qu)
    loss = loss_fn(y_pred , ans[0])
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  print(f"epoch {epoch} loss {train_loss/len(dataload)}")


def predict(model , question , threshold=0.5):
  numerical_question = text_to_index(question , vocab)
  question_tensor = torch.tensor(numerical_question).unsqueeze(0)
  output = model(question_tensor)
  probabilities = torch.softmax(output , dim=1)
  val , index = torch.max(probabilities , dim=1)
  if val < threshold:
    return "I don't know"
  else:
    print(list(vocab.keys())[index])

predict(model," Who discovered gravity?")
