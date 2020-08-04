import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import torch
import pandas as pd
from skimage.io import imread
from skimage.color import gray2rgb
import numpy as np
import torchvision as tv

train_mean = [0.59685254, 0.59685254, 0.59685254]
train_std = [0.16043035, 0.16043035, 0.16043035]


class ChallengeDataset(Dataset):
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        t = tv.transforms.Normalize((train_mean[0], train_mean[1], train_mean[2]), (train_std[0], train_std[1], train_std[2]))
        if mode == 'train':
            self._transform = tv.transforms.Compose([tv.transforms.ToTensor(), t, tv.transforms.RandomVerticalFlip(p=0.5)])
        else:
            self._transform = tv.transforms.Compose([tv.transforms.ToTensor(), t])
        #self._transform = torch.
        return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        csvthing = self.data
        img_name = csvthing.iloc[index]
        img_name = img_name[0]
        randolist = img_name.split(';')
        #img_name = randolist[0]
        image = imread(randolist[0])
        image = gray2rgb(image)
        #image = image.reshape((3,300,300))
        #image = torch.tensor(image)
        image = self._transform(image)
        #image = image.self.trans
        sample = (image, torch.tensor((int(randolist[1]),int(randolist[2]))))
        return sample

