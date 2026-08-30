import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pathlib import Path
import zipfile
import os
from torchvision import models
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
import torch
from torch import nn
from torchvision import datasets
import random
from timeit import default_timer as timer
from tqdm.auto import tqdm
from sklearn.metrics import accuracy_score , classification_report

dir = Path("plant_data/")
image_path = dir/"pics"
if image_path.is_dir():
  print(f"{image_path} already exists ")
else:
  print(f"{image_path} cannot exist creating one ")
  image_path.mkdir(parents=True, exist_ok=True)

os.environ["KAGGLE_USERNAME"] = "hunaintahir"
os.environ["KAGGLE_KEY"]      = "KGAT_c3e7d0c18a5b83f486a4d6051d3a9e75"
!kaggle datasets download -d vipoooool/new-plant-diseases-dataset
!unzip new-plant-diseases-dataset.zip -d plant_disease

train_dir = Path("plant_disease/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/train")
test_dir  = Path("plant_disease/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)/valid")

print(train_dir.is_dir())  # should print True
print(test_dir.is_dir())   # should print True
len(list(train_dir.glob("*/*.JPG"))) , len(list(test_dir.glob("*/*.JPG")))
class_names = sorted(os.listdir(train_dir))
print(f"Number of classes: {len(class_names)}")
print(class_names)
weight = models.EfficientNet_B0_Weights.DEFAULT
model = models.efficientnet_b0(weights=weight)


model.classifier
# freezing the conv layers
for params in model.features.parameters():
  params.requires_grad = False
train_transform = transforms.Compose([

    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2,
                           contrast=0.2,
                           saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder(root = train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root= test_dir , transform=test_transforms)
len(train_dataset), len(test_dataset) , len(train_dataset.classes)
train_dataset.class_to_idx
train_dataloader = DataLoader(dataset= train_dataset , batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(dataset= test_dataset , batch_size=32 , shuffle=True, num_workers=0,pin_memory=True)
len(train_dataloader), len(test_dataloader)
tt=next(iter(train_dataloader))
tt[0].shape
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.classifier.parameters(),
                             lr=0.001)
def plot_random_images(dataset, class_names, n=9):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    random_indices = random.sample(range(len(dataset)), n)

    for i, idx in enumerate(random_indices):
        img, label = dataset[idx]

        mean = torch.tensor([0.485, 0.456, 0.406])
        std  = torch.tensor([0.229, 0.224, 0.225])

        img = img.permute(1, 2, 0)
        img = (img * std) + mean
        img = img.clip(0, 1)

        axes[i].imshow(img)
        axes[i].set_title(class_names[label],
                         fontsize=9,
                         fontweight="bold",
                         color="green")
        axes[i].axis("off")

    plt.suptitle("Plant Disease Images",
                 fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


plot_random_images(train_dataset, train_dataset.classes)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 38),

)
!pip install torchinfo
from torchinfo import summary
summary(model, input_size=(64, 3, 224, 224))
model=model.to(device)
def training(model , train_dataload , loss , optimizer , device):
    model.train()
    train_loss , train_acc = 0 ,0
    for batch , (X, y) in enumerate(train_dataload):
      X, y= X.to(device,non_blocking =True) , y.to(device, non_blocking =True)
      y_pred=model(X)
      los=loss(y_pred , y)
      train_loss+=los.item()
      acc = accuracy_score(y.cpu().detach().numpy(), torch.argmax(torch.softmax(y_pred, dim=1), dim=1 ) .cpu().detach().numpy())
      train_acc+=acc
      optimizer.zero_grad()
      los.backward()
      optimizer.step()
    train_loss/=len(train_dataload)
    train_acc/=len(train_dataload)
    return train_loss , train_acc
def testing(model, test_dataload, loss, device):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(test_dataload):
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            y_pred = model(X)
            los = loss(y_pred, y)
            test_loss += los.item()
            acc = accuracy_score(
                y.cpu().numpy(),
                torch.argmax(torch.softmax(y_pred, dim=1), dim=1).cpu().numpy()
            )
            test_acc += acc

        test_loss /= len(test_dataload)
        test_acc  /= len(test_dataload)
    return test_loss, test_acc
def final_training(model , train_dataload , test_dataload , loss , optimizer , device , epoches):
  results = {
      "train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc" : []
  }
  for epoch in tqdm(range(epoches)):
    train_loss , train_acc = training(model, train_dataload , loss , optimizer , device)
    test_loss , test_acc = testing(model , test_dataload , loss , device)
    print(f"epoch {epoch+1} | train_loss {train_loss:.4f} | train_acc {train_acc:.4f} | test_loss {test_loss:.4f} | test_acc {test_acc:.4f}")
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)
  return results


from torch.utils.data import Subset

small_train = Subset(train_dataset, range(5000))
small_test  = Subset(test_dataset, range(1000))

small_train_loader = DataLoader(
    small_train,
    batch_size=32,
    shuffle=True,
    num_workers=0
)
small_test_loader = DataLoader(
    small_test,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

weights = models.EfficientNet_B0_Weights.DEFAULT
model   = models.efficientnet_b0(weights=weights)

for param in model.features.parameters():
    param.requires_grad = False
model.classifier = nn.Sequential(
    nn.Dropout(p=0.2),
    nn.Linear(1280, 38))
model = model.to(device)
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=0.001
)
print(f"Test batches: {len(test_dataloader)}")
print(f"Test samples: {len(test_dataset)}")
print(f"Device: {device}")
print(f"Model device: {next(model.parameters()).device}")


X, y = next(iter(train_dataloader))
print(f"Data device: {X.device}")
!pip install GPUtil
import GPUtil
GPUtil.showUtilization()
results = final_training(model , small_train_loader,small_test_loader , loss_fn , optimizer, device , epoches=10)


print(f"Test batches: {len(test_dataloader)}")
print(f"Test samples: {len(test_dataset)}")
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

plot_loss_curves(results)
torch.save(model.state_dict(), "plant_disease_model.pth")

