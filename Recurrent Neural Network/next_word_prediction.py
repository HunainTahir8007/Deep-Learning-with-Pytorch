import requests

url      = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)

with open("shakespeare.txt", "w") as f:
    f.write(response.text)
with open("shakespeare.txt", "r") as f:
    text = f.read()
import re
def tokenize(text):
  text= text.replace("?","")
  text= text.lower()
  text = re.sub(r"[^a-z\s]", "", text)
  return text.split()
tokenize(text)
vocab = {"<UNK>": 0}
for word in tokenize(text):
  if word  not in vocab:
    vocab[word]= len(vocab)

vocab.keys()
list(vocab.items())[1]
def text_to_index(text , vocab):
    index_text = []
    for word in tokenize(text):
        if word in vocab:
            index_text.append(vocab[word])
        else:
            index_text.append(vocab["<UNK>"])
            # return [vocab.get(word, 0) for word in tokenize(text)]
    return index_text
idx=text_to_index(text , vocab)
# sequences
inputs = []
targets = []
seq_len= 10
for i in range(len(idx)-seq_len):
  inputs.append(idx[i:i+seq_len])
  targets.append(idx[i+seq_len])

# creating the dataset
from torch.utils.data import Dataset , DataLoader
import torch
class word_pred_dataset(Dataset):
  def __init__(self, inputs, targets):
    self.inputs= inputs
    self.targets= targets
  def __len__(self):
    return len(self.inputs)
  def __getitem__(self, index):
     x= torch.tensor(self.inputs[index] , dtype= torch.long)
     y= torch.tensor(self.targets[index] , dtype= torch.long)
     return x, y

dataset = word_pred_dataset(inputs, targets)
print(f"Total samples : {len(dataset)}")

x, y = dataset[20]
print(f"Input tensor  : {x}")
print(f"Target tensor : {y}")
# Daraloader
data_loader = DataLoader(dataset=dataset , batch_size=32 , shuffle= True)
len(data_loader)
len(vocab)
# bulding the model arctecture
from torch import nn

class pred_word_model(nn.Module):
  def __init__(self):
    super().__init__()
    self.embeeding = nn.Embedding(len(vocab) , 128 , padding_idx=0)
    self.lstm1 = nn.LSTM(128, 256, num_layers=1,
                             batch_first=True)

    self.drop = nn.Dropout(0.3)
    self.fc= nn.Sequential(
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, len(vocab))

    )
  def forward(self,x ):
    embeeded = self.embeeding(x)
    output1 , (hidden1 , cell1) = self.lstm1(embeeded)
    out = output1[:, -1, :]
    out = self.drop(out)
    out = self.fc(out)
    return out


test_input = torch.randint(0, len(vocab), (1, 10))
print(f"Input shape  : {test_input.shape}")
model = pred_word_model()
output = model(test_input)
print(f"Output shape : {output.shape}")

device = "cuda" if torch.cuda.is_available() else "cpu"
device
model = pred_word_model().to(device)
!pip install torchinfo

from torchinfo import summary
import torch
summary(model , input_size=(32, 10), dtypes=[torch.long])
# loss + optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters() , lr=0.001)
from tqdm.auto import tqdm
from timeit import default_timer as timer
from sklearn.metrics import accuracy_score
# training the model
epoches  = 50
start = timer()
for epoch in tqdm(range(epoches)):
  model.train()
  train_loss , train_acc = 0 , 0
  for X , y in data_loader:
    X= X.to(device)
    y= y.to(device)
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred , y)
    train_loss += loss.item()
    acc = accuracy_score(y.cpu().numpy() , torch.argmax(y_pred , dim=1) .cpu().numpy())
    train_acc += acc
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
  train_loss = train_loss / len(data_loader)
  train_acc = train_acc / len(data_loader)
  end = timer()

  print(f"Epoch {epoch + 1 } | train_loss {train_loss :.4f} | train_accuracy {train_acc:.4f}")
  print(f"Time taken : {end - start}")
torch.save(model.state_dict(), "next_word_model.pth")
print("Model saved ")
from google.colab import files
files.download("next_word_model.pth")

def generate_text(model, start_text, vocab, seq_len=10, gen_len=50, device="cpu"):

    model.eval()
    idx_to_word = {v: k for k, v in vocab.items()}
    words = start_text.lower().split()
    input_seq = [vocab.get(word, vocab["<UNK>"]) for word in words]
    generated = input_seq.copy()
    for _ in range(gen_len):
        x = torch.tensor([generated[-seq_len:]], dtype=torch.long).to(device)
        with torch.no_grad():
            y_pred = model(x)
            next_idx = torch.argmax(y_pred, dim=1).item()
        generated.append(next_idx)
    generated_text = " ".join([idx_to_word[i] for i in generated])
    return generated_text

start_text = "what will the "
gen_len = 10
seq_len = 10
generated = generate_text(model, start_text, vocab, seq_len=seq_len, gen_len=gen_len, device=device)
print(generated)
