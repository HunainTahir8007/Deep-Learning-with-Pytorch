import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
# Training data
train_data = datasets.FashionMNIST(
    root="fashion" ,
    train= True ,
    download= True ,
    transform= ToTensor()
)
# tesing data
test_data = datasets.FashionMNIST(
    root="fashion_test" ,
    train= False,
    download= True ,
    transform= ToTensor()

)
len(train_data) , len(test_data)
print("train classes",train_data.classes)
print("train IDX ",train_data.class_to_idx)
print("train targets" , train_data.targets)

image , label=train_data[1]
image , label
# Showing image
image , label =train_data[0]
plt.imshow(image.squeeze())
plt.title(train_data.classes[label])
plt.axis(False)
plt.show()
torch.manual_seed(42)
fig=plt.figure(figsize=(10,10))
rows , cols = 4 , 4
for i in range(1 , rows * cols +1 ):
  random_index = torch.randint(0 , len(train_data),size= [1]).item()
  image  ,  label = train_data[random_index]
  fig.add_subplot(rows , cols , i )
  plt.imshow(image.squeeze())
  plt.title(train_data.classes[label])
  plt.axis(False)
from torchvision.datasets.vision import data
 # loading  the data
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data , batch_size= 32 , shuffle= True)
test_dataloader = DataLoader( test_data , batch_size= 32 , shuffle= True)

print(f"lenth of the train dataloader {len(train_dataloader)} with batch size {train_dataloader.batch_size}")
print(f"lenth of the test dataloader {len(test_dataloader)} with   batch size {test_dataloader.batch_size}"  )

train_feature_batch , train_label_batch = next(iter(train_dataloader))
train_feature_batch.shape , train_label_batch.shape
#torch.manual_seed(42)
random_idx = torch.randint(0 , len(train_feature_batch), size= [1]).item()
image , label = train_feature_batch[random_idx], train_label_batch[random_idx]
plt.axis(False)
plt.imshow(image.squeeze(), cmap='grey')
plt.title(train_data.classes[label])
plt.show()

# work of flatten layer
flatten_mod= nn.Flatten()
x= train_feature_batch[0]
lay= flatten_mod(x)
lay.shape , x.shape  # flatten layer just multiply the height * width


# create the model
class fashionmodel(nn.Module):
  def __init__(self,input_shape: int  , hidden_shape: int  , output_shape : int  ) :
    super().__init__()
    self.layer_stack = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=input_shape , out_features=hidden_shape),
        nn.Linear(in_features=hidden_shape , out_features= output_shape)
    )
  def forward(self , x):
      return self.layer_stack(x)

torch.manual_seed(42)
model_1 = fashionmodel(input_shape=28 * 28 , hidden_shape=10 , output_shape=10).to("cpu")
model_1
dummy = torch.rand(1,1, 28, 28)
model_1(dummy)
model_1.state_dict()
#loss + optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params= model_1.parameters(), lr= 0.1)

 # timer
from timeit import default_timer as timer
def print_timer(start: float , end: float , device: torch.device = None):
  total_time= start - end
  print(f"Train time on {device} : {total_time:.3f} seconds")
  return total_time
epochs = 3
from tqdm.auto import tqdm
torch.manual_seed(42)
for epoch in tqdm(range(epochs)):

    train_loss = 0
    model_1.train()

    for X, y in train_dataloader:
        y_pred = model_1(X)
        loss = loss_fn(y_pred, y)

        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    # Testing
    test_loss = 0
    test_acc = 0
    model_1.eval()

    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_pred = model_1(X_test)

            test_loss += loss_fn(test_pred, y_test).item()
            test_acc += accuracy_score(
                y_test.cpu(),
                test_pred.argmax(dim=1).cpu()
            )

    test_loss /= len(test_dataloader)
    test_acc /= len(test_dataloader)

    print(f"Epoch {epoch+1} | "
          f"Train loss: {train_loss:.4f} | "
          f"Test loss: {test_loss:.4f} | "
          f"Test acc: {test_acc:.4f}")
# model evaluation
acc, loss = 0 , 0

model_1.eval()
with torch.inference_mode():
  for X_test , Y_test in test_dataloader:
    test_pred  =  model_1(X_test)
    loss += loss_fn(test_pred , Y_test)
    acc += accuracy_score(Y_test , test_pred.argmax(dim=1))

  avg_acc = acc/ len(test_dataloader)
  avg_loss = loss/ len(test_dataloader)

print(f"model accuracy {avg_acc}")
print(f"model loss {avg_loss}")

# device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
# model with gpu
model_2 = nn.Sequential(
    nn.Flatten(),
    nn.Linear(in_features=28 * 28 , out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10 , out_features=10),
    nn.ReLU()
).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params= model_2.parameters(), lr= 0.1)
# traing loop for not linear
start_time = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
  model_2.train()
  train_loss = 0
  for X, y in train_dataloader:
    X= X.to(device)
    y= y.to(device)
    y_pred = model_2(X)
    loss = loss_fn(y_pred , y )
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss = train_loss / len(train_dataloader)

  model_2.eval()
  with torch.inference_mode():
    test_loss_epoch = 0
    test_acc_epoch = 0
    for X , y in test_dataloader:
      X= X.to(device)
      y= y.to(device)
      test_pred = model_2(X)
      loss = loss_fn(test_pred , y)
      test_loss_epoch += loss.item()

      test_acc_epoch += accuracy_score(y.cpu() ,test_pred.argmax(dim=1).cpu())
    test_loss_epoch = test_loss_epoch / len(test_dataloader)
    test_acc_epoch = test_acc_epoch / len(test_dataloader)

    print(f"Epoch {epoch+1} | "
          f"Train loss: {train_loss:.4f} | "
          f"Test loss: {test_loss_epoch:.4f} | "
          f"Test acc: {test_acc_epoch:.4f}")
    end_time = timer()
  print(end_time - start_time)
from torch.nn.modules import padding

class fashionmodel2(nn.Module):
  def __init__(self,input_shape: int  , hidden_shape: int  , output_shape : int  ) :
    super().__init__()
    self.covt_block1= nn.Sequential(
        nn.Conv2d(in_channels=input_shape , out_channels=hidden_shape , kernel_size=3 , stride=1 , padding= 1 ),
        nn.ReLU(),
        nn.Conv2d(in_channels = hidden_shape , out_channels= hidden_shape , kernel_size = 3 , stride =1 , padding= 1 ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2  )
        )
    self.covt_block2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_shape , out_channels= hidden_shape , kernel_size=3 , padding=1 ),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_shape , out_channels= hidden_shape , kernel_size=3 ,padding=1 ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features = hidden_shape * 7 * 7, out_features= output_shape)

    )
  def forward(self , x):
      x=self.covt_block1(x)

      x= self.covt_block2(x)

      x= self.classifier(x)

      return x

torch.manual_seed(42)
model_3 = fashionmodel2(input_shape=1 , hidden_shape=10 , output_shape=10).to(device)

torch.manual_seed(42)
images = torch.randn(size=(32 , 3 , 64 ,64 ))
test_image= images[0]
images.shape , test_image.shape , test_image
single= nn.Conv2d(in_channels=3 , out_channels=10 , kernel_size = 3 , stride=1 , padding = 0 )
single(test_image)
rand_tensor = torch.randn(size=(1,28, 28))
rand_tensor.shape
model_3(rand_tensor.unsqueeze(0).to(device))
# loss  + optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params = model_3.parameters(), lr=0.1 )
start_time = timer()
epochs = 3
for epoch in tqdm(range(epochs)):
  model_3.train()
  train_loss = 0
  for X, y in train_dataloader:
    X= X.to(device)
    y= y.to(device)
    y_pred = model_2(X)
    loss = loss_fn(y_pred , y )
    train_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  train_loss = train_loss / len(train_dataloader)

  model_2.eval()
  with torch.inference_mode():
    test_loss_epoch = 0
    test_acc_epoch = 0
    for X , y in test_dataloader:
      X= X.to(device)
      y= y.to(device)
      test_pred = model_2(X)
      loss = loss_fn(test_pred , y)
      test_loss_epoch += loss.item()

      test_acc_epoch += accuracy_score(y.cpu() ,test_pred.argmax(dim=1).cpu())
    test_loss_epoch = test_loss_epoch / len(test_dataloader)
    test_acc_epoch = test_acc_epoch / len(test_dataloader)

    print(f"Epoch {epoch+1} | "
          f"Train loss: {train_loss:.4f} | "
          f"Test loss: {test_loss_epoch:.4f} | "
          f"Test acc: {test_acc_epoch:.4f}")
    end_time = timer()
  print(end_time - start_time)

