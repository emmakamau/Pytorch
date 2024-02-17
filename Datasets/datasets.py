"""
Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code to be decoupled
from our model training code for better readability and modularity. PyTorch provides two data primitives:
torch.utils.data.DataLoader and torch.utils.data.Dataset that allow you to use pre-loaded datasets as well as your
own data. Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset
to enable easy access to the samples.

    1. root is the path where the train/test data is stored,
    2. train specifies training or test dataset,
    3. download=True downloads the data from the internet if it’s not available at root.
    4. transform and target_transform specify the feature and label transformations

    Dataset used => https://github.com/zalandoresearch/fashion-mnist
"""

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def training_data_():
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )
    return training_data


def test_data_():
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    return test_data


# We are using matplotlib for this
def visualize_data():
    training_data = training_data_()
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
        sample_idx = torch.randint(len(training_data), size=(1,)).item()
        img, label = training_data[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()

visualize_data()
