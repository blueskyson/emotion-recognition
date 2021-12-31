import pandas as pd
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = np.array(self.images[idx])
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx]).type(torch.long)
        sample = (img, label)

        return sample


def load_data(path="fer2013.csv"):
    fer2013 = pd.read_csv(path)
    emotion_mapping = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Sad",
        5: "Surprise",
        6: "Neutral",
    }
    return fer2013, emotion_mapping


def prepare_data(data):
    """ Prepare data for modeling
        input: data frame with labels und pixel data
        output: image and label array """

    image_array = np.zeros(shape=(len(data), 48, 48))
    image_label = np.array(list(map(int, data["emotion"])))

    for i, row in enumerate(data.index):
        image = np.fromstring(data.loc[row, "pixels"], dtype=int, sep=" ")
        image = np.reshape(image, (48, 48))
        image_array[i] = image

    return image_array, image_label


def get_dataloaders(path="fer2013.csv", bs=16):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation

        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013["Usage"] == "Training"])
    print(xtrain[0])
    xval, yval = prepare_data(fer2013[fer2013["Usage"] == "PrivateTest"])
    xtest, ytest = prepare_data(fer2013[fer2013["Usage"] == "PublicTest"])

    mu, st = 0, 255
    test_transform = transforms.Compose(
        [
            transforms.Resize(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,)),
        ]
    )

    train = CustomDataset(xtrain, ytrain, test_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=bs, shuffle=True, num_workers=0)
    testloader = DataLoader(test, batch_size=bs, shuffle=True, num_workers=0)

    return trainloader, valloader, testloader, emotion_mapping

def custom_imageloader(orig_img):
    mu, st = 0, 255
    test_transform = transforms.Compose(
        [
            transforms.Resize(40),
            transforms.ToTensor(),
            #transforms.Normalize(mean=(mu,), std=(st,)),
        ]
    )
    img = test_transform(orig_img)
    return img