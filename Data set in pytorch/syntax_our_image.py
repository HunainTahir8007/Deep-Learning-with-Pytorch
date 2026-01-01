import pandas as pd 
import os 
from torchvision.io import decode_image
from torch.utils.data import dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 


class custom_image_dataset(dataset):
    def __init__(self,anotation_file,img_dir,transform=None,target_transform=None):
            self.img_labels=pd.read_csv()
            self.img_dir=img_dir
            self.transform=transform
            self.target_transform=target_transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __get_items__(self,idx):
       img_path=os.path.join(self.img_dir,self.img_labels.iloc[idx,0])
       image=decode_image(img_path)
       labels=self.img_labels.iloc[idx,1]
       if self.transform:
           image=self.transform(image)
       if self.target_transform:
            labels=self.target_transform(image)
       return image,labels
   

train_loader=DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
# Display image and label.

train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")

img = train_features[0].squeeze()
label = train_labels[0]

plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
