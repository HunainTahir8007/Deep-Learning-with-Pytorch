import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2470, 0.2435, 0.2616])
])
train_data = datasets.CIFAR10(root="cifar_train",
 train=True , transform=train_transform, download= True)
test_data = datasets.CIFAR10(root= "cifar_test" , train=False , transform=test_transform , download=True)

image , label = train_data[1]
plt.imshow(image.permute(1,2,0))
plt.title(train_data.classes[label])
plt.axis(False)
len(train_data) , len(test_data)
# plotting the random images
torch.manual_seed(42)
fig = plt.figure(figsize = (10 ,10 ))
rows , cols = 4 ,4
for i in range(1, rows * cols +1 ):
  rand_idx = torch.randint(0 , len(train_data), size = [1]).item()
  image , label = train_data[rand_idx]
  fig.add_subplot(rows, cols, i)
  plt.imshow(image.permute(1,2 ,0))
  plt.title(train_data.classes[label])
  plt.axis(False)

train_dataloader = DataLoader(dataset=train_data , batch_size = 64 , shuffle=True)
test_dataloader = DataLoader(dataset = test_data , batch_size= 64 , shuffle= True)


len(train_data)/ len(train_dataloader)
from torch.nn.modules.pooling import MaxPool2d
# model
class cifarModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
        nn.Conv2d(3 , 64 , kernel_size=3 , padding= 1, stride=1 ),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2 ),

        nn.Conv2d(64 , 128 , kernel_size=3 , padding=1, stride=1 ),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(128 , 256 , kernel_size = 3 , stride = 1  , padding =1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),

        nn.Conv2d(256 , 128 , kernel_size = 3 , stride = 1 , padding = 1 ),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size =2 ),
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 10)

    )
  def forward(self , x ):
    x= self.conv_block_1(x)

    x= self.conv_block_2(x)

    x= self.classifier(x)
    return x
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
!nvidia-smi
model= cifarModel().to(device)
rand_tensor = torch.randn(size=(1,3,32, 32)).to(device)
rand_tensor.shape
testtt= model(rand_tensor)
print(testtt.shape)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params= model.parameters() , lr = 0.001 )
from timeit import default_timer as timer
from tqdm.auto import tqdm


from IPython.paths import ensure_dir_exists
# set up traing loop
epochs = 20
start_time = timer()
for epoch in  tqdm(range(epochs)):
  train_acc = 0
  train_loss = 0
  model.train()
  for batch , (X , y ) in enumerate(train_dataloader):
    X=X.to(device)
    y= y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred , y )
    acc = accuracy_score(
    y_pred.cpu().detach().argmax(dim=1).numpy(),
    y.cpu().numpy()
)
    train_acc = train_acc + acc
    train_loss = train_loss + loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss = train_loss / len(train_dataloader)
  train_acc = train_acc / len(train_dataloader)
  end_time = timer()
  print(f"Epoch : {epoch} |  train loss : {train_loss} | train_accuracy {train_acc}")
  print(end_time- start_time)

model.eval()
with torch.inference_mode():
    test_acc  = 0
    test_loss = 0

    for batch, (X, y) in enumerate(test_dataloader):
        X, y = X.to(device), y.to(device)

        test_pred = model(X)
        loss = loss_fn(test_pred, y)
        test_loss += loss.item()
        test_acc  += accuracy_score(
            test_pred.cpu().argmax(dim=1).numpy(),
            y.cpu().numpy()
        )

    test_loss /= len(test_dataloader)
    test_acc  /= len(test_dataloader)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")




