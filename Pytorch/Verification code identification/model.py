import numpy as np
import random
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.optim
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
import torch.nn.functional as F
from torchvision import transforms as T
import tensorflow as tf


data = pd.read_csv("qwe.csv")

label = data.loc[:,data.columns!="ID"].values

path = data["ID"].values

transform = T.Compose([
   # T.Resize((128,128)),
    T.ToTensor(),
    T.Normalize(std=[0.5],mean=[0.5])
])


class DataSet(Dataset):
    def __init__(self,root,label,transform):
        self.image_path = root
        self.label = label
        self.transform = transform

    def __getitem__(self, index):
        path = self.image_path[index]
        img = Image.open(path)

        img = self.transform(img)


        label = self.label[index]

        a = tf.one_hot(label[0], depth=10)

        a = a.numpy()
        for i in label[1:]:
            b = tf.one_hot(i, depth=10)
            b = b.numpy()

            a = np.concatenate((a, b), axis=0)

        return img,a

    def __len__(self):
        return len(self.image_path)

dataset = DataSet(path,label,transform)

data_loader = DataLoader(dataset,batch_size=64,shuffle=True,drop_last=True)


class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)


        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),  # 30 80

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128,256,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),   # 15 40

        )

        self.fc = nn.Sequential(
            nn.Linear(256 * 15 * 40, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 40)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = F.softmax(x,dim=-1)
        return x




model = CNN_Network()

model.train()
optimizer = torch.optim.Adam(model.parameters(),lr=0.002)

error = nn.MultiLabelSoftMarginLoss()



for i in range(5):
    for (batch_x,batch_y) in data_loader:

        optimizer.zero_grad()
        image = Variable(batch_x)
        label = Variable(batch_y)

        out = model(image)

        loss = error(out,label)

        print(loss)

        loss.backward()
        optimizer.step()

torch.save(model.state_dict(),"model.pth")






