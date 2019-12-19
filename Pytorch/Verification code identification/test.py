import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2
import numpy as np




class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)


        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(stride=2, kernel_size=2),  # 30 80

        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),   # 15 40

        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 15 * 40, 2048),
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

        return x


cnn = CNN_Network()
cnn.load_state_dict(torch.load("验证码识别.pth"))


a = cv2.imread("./data/9354.jpg",0)
b = cv2.resize(a,(200,200))
cv2.imshow('a',b)
cv2.waitKey(0)
a = a/255.
a = torch.from_numpy(a).float()

a = torch.unsqueeze(a,0)
a = torch.unsqueeze(a,0)
pred = cnn(a)
print(pred.size())
a1 = torch.argmax(pred[0,:10],dim=0)
a2 = torch.argmax(pred[0,10:20],dim=0)
a3 = torch.argmax(pred[0,20:30],dim=0)
a4 = torch.argmax(pred[0,30:],dim=0)

pred = [a1,a2,a3,a4]
print(pred)