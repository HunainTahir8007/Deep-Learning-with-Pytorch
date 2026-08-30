import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from torchvision import datasets , transforms
from torch.utils.data import dataloader
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
# collecting data afrom git hub
import requests
import zipfile
from pathlib import Path

data_path = Path("data/")
image_path = data_path/ "pizza_steak_sushi"
if image_path.is_dir():
  print(f"{image_path} already exists ")
else:
  print(f"{image_path} cannot exist creating one ")
  image_path.mkdir(parents=True, exist_ok=True)

# downloading data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
  request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/refs/heads/main/data/pizza_steak_sushi.zip")
  print("Downloading data")
  f.write(request.content)

#Unzip the data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
  print("Unzipping the data")
  zip_ref.extractall(image_path)


# cheaking the data
import os
def walk_through_dir(dir_path):
  for dirpath , dirnames , filenames in os.walk(dir_path):
    print(f"directories {len(dirnames)} and images {len(filenames)}  in folder {dirpath} ")

walk_through_dir(image_path)
# train and testing paths
train_dir = image_path / "train"
test_dir  = image_path / "test"

# visualizing the random picture
from PIL import Image
import random

image_path_list = list(image_path.glob("*/*/*.jpg"))

random_image_path = random.choice(image_path_list)
print(random_image_path)
print(random_image_path.parent.stem)
img = Image.open(random_image_path)
print("image height " , img.height)
print("image width " , img.width)
img
# visualizing by using the matplotlib
img_arr = np.asarray(img)
plt.imshow(img_arr)
plt.title(f"Image shape {img_arr.shape}")
plt.axis(False)
# COnverting the images into tensors
data_transform = transforms.Compose([
    transforms.Resize((64, 64 )),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(0.5),

])
data_transform(img)
def plot_transformed_images(image_paths : list , transform ,n =3 , seed = None):
  if seed:
    random.seed(seed)
  random_image_paths = random.sample(image_paths,k=n)
  for image_path in random_image_paths:
    with Image.open(image_path) as f:
      fig , ax = plt.subplots(1,2)
      ax[0].imshow(f)
      ax[0].set_title(f"Original Image\nSize : {f.size}")
      ax[0].axis(False)

      transformed_image = transform(f).permute(1,2,0)
      ax[1].imshow(transformed_image)
      ax[1].set_title(f"Transformed Image\nSize : {transformed_image.shape}")
      ax[1].axis(False)

      fig.suptitle(f"Class : {image_path.parent.stem}",fontsize = 16)

plot_transformed_images(image_paths= image_path_list , transform=data_transform , n=3 , seed = 42)

train_data = datasets.ImageFolder(root = train_dir,
                                  transform=data_transform,
                                  target_transform=None)
test_data = datasets.ImageFolder(root= test_dir , transform=data_transform , target_transform=None)
train_data.class_to_idx
train_data.classes
len(train_data) , len(test_data)
train_data.targets
img , label =train_data[0]
img.shape , label
img_permute = img.permute(1,2,0)
plt.figure(figsize=(10,7))
plt.imshow(img_permute)
plt.axis(False)
plt.title([label])
#  data loader
train_dataLoader = dataloader.DataLoader(dataset=train_data , batch_size=1 , num_workers=1 , shuffle=True)
test_dataLoader = dataloader.DataLoader(dataset=test_data , batch_size=1 , shuffle=False , num_workers=1)
import os
import pathlib
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple , Dict , List

# cheaking the number of classes in folder
target_directory = train_dir

class_names = sorted(entry.name for entry in list(os.scandir(target_directory)) )
class_names , train_data.class_to_idx
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:


    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")


    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx
# custom dataset
class ImageFolderCustom(Dataset):
    def __init__(self , targ_dir : str , transform = None) :
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(targ_dir)
    def load_image(self, index: int) -> Image.Image:
        "Loads and returns PIL image at index"
        image_path = self.paths[index]
        return Image.open(image_path)
    def __len__(self) -> int:
        "Returns the total number of samples"
        return len(self.paths)
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        "Returns one sample of data, data and label(target)"
        img = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(img), class_idx
        else:
            return img, class_idx
# transforms
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])
train_data_custom = ImageFolderCustom(targ_dir=train_dir, transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir, transform=test_transforms)
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):

    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")

    if seed:
        random.seed(seed)
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16, 8))
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]
        targ_image_adjust = targ_image.permute(1, 2, 0)
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)
display_random_images(train_data , n=5 , classes=class_names , seed= None)
display_random_images(train_data_custom , n=5 , classes=class_names , seed= None)
from torch.utils.data import DataLoader
train_dataloader_custom = DataLoader(dataset=train_data_custom,
                                     batch_size=1,
                                     num_workers=0,
                                     shuffle=True)

test_dataloader_custom = DataLoader(dataset=test_data_custom,
                                    batch_size=1,
                                    num_workers=0,
                                    shuffle=False)

train_dataloader_custom, test_dataloader_custom
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
image_path_list = list(image_path.glob("*/*/*.jpg"))


plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None
)
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])
from torchvision import datasets
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)
import os
from torch.utils.data import DataLoader

BATCH_SIZE = 32
NUM_WORKERS = os.cpu_count()
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

train_dataloader_simple, test_dataloader_simple
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=hidden_units * 13 * 13,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)

        return x


torch.manual_seed(42)
model_0 = TinyVGG(input_shape=3,
                  hidden_units=10,
                  output_shape=len(train_data.classes))
model_0
device = "cuda" if torch.cuda.is_available() else 'cpu'
device
torch.manual_seed(42)
model_0 = TinyVGG(input_shape= 3 , hidden_units= 10 , output_shape= len(class_names)).to(device)

# giving the single shape to cheak the hidden units of nn.flatten
imgg , labb = next(iter(train_dataloader_simple))
imgg.shape , labb.shape
model_0(imgg.to(device))
# summary of the model
!pip install torchinfo
from torchinfo import summary
summary(model_0 , input_size=(1,3,64,64))
# training and testing
def train_model(model : torch.nn.Module ,
                dataloader : torch.utils.data.DataLoader,
                loss_fn : torch.nn.Module ,
                optimizer : torch.optim.Optimizer ,
                device = device):
  model_0.train()
  train_loss , train_acc = 0 ,0

  for batch, (X,y) in enumerate(dataloader):
    X=X.to(device)
    y=y.to(device)
    y_pred = model(X)
    loss = loss_fn(y_pred , y)
    train_loss+=loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    y_pred_class = torch.argmax(torch.softmax(y_pred , dim=1), dim=1)
    train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss , train_acc




def testing(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    model.eval()
    test_loss, test_acc = 0, 0
    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            test_pred_logits = model(X)
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc
def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):

    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }


    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_model(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = testing(model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn)


        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )


        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)


    return results
torch.manual_seed(42)
torch.cuda.manual_seed(42)

num_epoches = 50
optimizer = torch.optim.Adam(params=model_0.parameters() , lr=0.001)
loss_fn = nn.CrossEntropyLoss()
from timeit import default_timer as timer
start_time = timer()
model_0_results = train(model=model_0 ,train_dataloader=train_dataloader_simple ,
                       test_dataloader=  test_dataloader_simple,
                        optimizer= optimizer ,
                       loss_fn=loss_fn ,
                        epochs= num_epoches)
end_time = timer()
print(f"Total training time : {end_time - start_time}")





