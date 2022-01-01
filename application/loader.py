# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 15:19:54 2022

@author: USER
"""

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
    emotion_mapping2 = {
        0: "Angry",
        1: "Disgust",
        2: "Fear",
        3: "Happy",
        4: "Neutral",
        5: "Sad",
        6: "Surprise",
    }
    return fer2013, emotion_mapping, emotion_mapping2


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

    fer2013, emotion_mapping, emotion_mapping2 = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013["Usage"] == "Training"])
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

    return trainloader, valloader, testloader, emotion_mapping, emotion_mapping2

def get_dataloaders2(path="fer2013.csv", bs=16):
    """ Prepare train, val, & test dataloaders
        Augment training data using:
            - cropping
            - shifting (vertical/horizental)
            - horizental flipping
            - rotation
        input: path to fer2013 csv file
        output: (Dataloader, Dataloader, Dataloader) """

    fer2013, emotion_mapping, emotion_mapping2 = load_data(path)

    xtrain, ytrain = prepare_data(fer2013[fer2013["Usage"] == "Training"])
    xval, yval = prepare_data(fer2013[fer2013["Usage"] == "PrivateTest"])
    xtest, ytest = prepare_data(fer2013[fer2013["Usage"] == "PublicTest"])

    MEAN, STD = (0.485), (0.229)
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        normalize,
    ])

    train = CustomDataset(xtrain, ytrain, test_transform)
    val = CustomDataset(xval, yval, test_transform)
    test = CustomDataset(xtest, ytest, test_transform)

    trainloader = DataLoader(train, batch_size=bs, shuffle=True, num_workers=0)
    valloader = DataLoader(val, batch_size=bs, shuffle=True, num_workers=0)
    testloader = DataLoader(test, batch_size=bs, shuffle=True, num_workers=0)

    return trainloader, valloader, testloader


def custom_imageloader(xtest):
    height = xtest.shape[0]
    width = xtest.shape[1]
    image_array = np.zeros(shape=(1, height, width))
    image_array[0] = xtest
    mu, st = 0, 255
    test_transform = transforms.Compose(
        [
            transforms.Resize(40),
            transforms.ToTensor(),
            transforms.Normalize(mean=(mu,), std=(st,)),
        ]
    )
    test = CustomDataset(image_array, [-1], test_transform)
    testloader = DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
    return testloader

def custom_imageloader2(xtest):
    height = xtest.shape[0]
    width = xtest.shape[1]
    image_array = np.zeros(shape=(1, height, width))
    image_array[0] = xtest

    MEAN, STD = (0.485), (0.229)
    normalize = transforms.Normalize(mean=MEAN, std=STD)
    test_transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(48),
        transforms.CenterCrop(48),
        transforms.ToTensor(),
        normalize,
    ])
    test = CustomDataset(image_array, [-1], test_transform)
    testloader = DataLoader(test, batch_size=1, shuffle=True, num_workers=0)
    return testloader
