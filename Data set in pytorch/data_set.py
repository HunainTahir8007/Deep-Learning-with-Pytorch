import torch 
from torch.utils.data import dataset
from  torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt 

training_dataset=datasets.FashionMNIST(
    root='Fasion_data',
    download=True,
    train=True,
    transform=ToTensor()

)
test_data = datasets.FashionMNIST(
    root='Fasion_data',
    train=False,
    download=True,
    transform=ToTensor()
)
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_dataset), size=(1,)).item()
    img, label = training_dataset[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()