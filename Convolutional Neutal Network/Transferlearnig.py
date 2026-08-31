import random
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader , random_split
from torchvision import datasets,transforms
import torch
from torch import nn
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import InterpolationMode
from sklearn.metrics import confusion_matrix , accuracy_score
from tqdm.auto import tqdm
from timeit import default_timer as timer
from pathlib import Path
from torchvision import models

# !pip install ultralytics    work on the collab 
# from ultralytics import YOLO

os.environ["KAGGLE_USERNAME"] = "hunaintahir"
os.environ["KAGGLE_KEY"] = ""

device = "cuda" if torch.cuda.is_available() else "cpu"

#  Now we use the transfer learning to train the model to cheak the model performance

# setting up model
weight = models.EfficientNet_B0_Weights.DEFAULT
model_1 = models.efficientnet_b0(weights=weight)

# freezing the feature extraction layers
for param in model_1.features.parameters():
  param.requires_grad=False
# Transformation For the Transfer model
train_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2 , 0.2 ,0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Lambda(lambda x: x.convert("RGB")),
     transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
# creating thw image foldder for required transforms

train_data = datasets.ImageFolder(root = "vehicle_split/train" , transform= train_transforms )
test_data = datasets.ImageFolder(root ="vehicle_split/val", transform=test_transforms )
len(train_data) , len(test_data)
model_1.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(in_features=1280,
              out_features=4)
)
model_1.classifier
model_1 =model_1.to(device)


def plot_loss_curves(results):
    epochs = range(len(results["train_loss"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, results["train_loss"], label="train_loss")
    ax1.plot(epochs, results["test_loss"],  label="test_loss")
    ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)

    ax2.plot(epochs, results["train_acc"], label="train_acc")
    ax2.plot(epochs, results["test_acc"],  label="test_acc")
    ax2.set_title("Accuracy"); ax2.legend(); ax2.grid(True)

    plt.tight_layout()
    plt.show()



# setting up Dataloader
train_dataloader = DataLoader(train_data , batch_size=32 , shuffle=True)
test_dataloader = DataLoader(test_data , batch_size=32 , shuffle=False)
# setting up loss and optimizer
def train_step(model , dataload , loss_fn , optimizer , device):
  model.train()
  train_loss , train_acc = 0,0
  for batch , (X,y) in enumerate(dataload):
    X= X.to(device)
    y=y.to(device)
    y_pred = model(X)
    loss= loss_fn(y_pred , y)
    train_loss += loss.item()
    acc = accuracy_score( y.cpu(), torch.argmax(torch.softmax(y_pred.cpu(), dim=1), dim=1) )
    train_acc += acc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  train_loss= train_loss /len(dataload)
  train_acc= train_acc /len(dataload)
  return train_loss , train_acc


def test_step(model, dataload , loss_fn , device):
  model.eval()
  test_loss, test_acc = 0 , 0
  with torch.inference_mode():
    for batch , (X,y) in enumerate(dataload):
      X= X.to(device)
      y=y.to(device)
      test_pred= model(X)
      loss = loss_fn(test_pred, y )
      test_loss+= loss.item()
      acc= accuracy_score( y.cpu() , torch.argmax(torch.softmax(test_pred.cpu(), dim=1 ), dim=1 ))
      test_acc += acc
    test_loss = test_loss / len(dataload)
    test_acc = test_acc / len(dataload)
    return test_loss , test_acc

def training(model , train_dataload , test_dataload , optimizer , loss_fn , epochs , device):
  results = {
      "train_loss" :[] ,
      "train_acc"  :[] ,
      "test_loss"  :[] ,
      "test_acc"   :[]
  }
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model , train_dataload ,loss_fn , optimizer , device )
    test_loss , test_acc = test_step(model , test_dataload , loss_fn , device)
    print(f"epoch {epoch+1} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_loss {test_loss:.4f} | test_acc {test_acc:.4f}")
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  return results



loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model_1.classifier.parameters(), lr = 0.001)
res_mod1=training(model_1,train_dataloader , test_dataloader , optimizer , loss_fn , 5 , device )
plot_loss_curves(res_mod1)


yolo_model = YOLO("yolov8n.pt")
all_images = list(Path("vehicle_split/val").glob("*/*.jpg")) + \
             list(Path("vehicle_split/val").glob("*/*.JPG")) + \
             list(Path("vehicle_split/val").glob("*/*.png")) + \
             list(Path("vehicle_split/val").glob("*/*.PNG")) + \
             list(Path("vehicle_split/val").glob("*/*.jpeg"))



random_images = random.sample(all_images, 10)
result = yolo_model.predict(source=random_images,classes=[0,2, 3, 5, 7], conf=0.5)
for r in result[:10]:
    r.show()
    
    


uploaded = files.upload()

image_path = list(uploaded.keys())[0]

result = yolo_model.predict(
    source=image_path,
    conf=0.5
)

result[0].show()
