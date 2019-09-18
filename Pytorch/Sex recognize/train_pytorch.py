import torch
from torch.autograd import Variable
import torch.optim
import torch.utils.data as data
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn.functional as F
import torch.nn
from torchvision import transforms as T
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import os
import time
transform = T.Compose([
    T.Resize(128),
    T.CenterCrop(128),
    T.ToTensor(),
    T.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])



class dataset(Dataset):
    def __init__(self,root,transform=None):
        imgs = os.listdir(root)

        self.imgs = [os.path.join(root,img) for img in imgs]
        self.transform = transform
    def __getitem__(self, index):
        img_path = self.imgs[index]

        label = 0 if 'woman' in img_path.split('/')[-1] else 1

        data = Image.open(img_path)
        data = self.transform(data)

        return data,label
    def __len__(self):
        return len(self.imgs)


data = dataset('./man',transform=transform)




data = DataLoader(data,batch_size=128,shuffle=True,drop_last=True,num_workers=0)

class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2) #64
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2,stride=2) #32
        )

        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2,stride=2)  #16
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128*16*16,2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048,2048),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(2048,2)
        )

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = F.softmax(self.fc(x),dim=1)
        return x



cnn = CNN()
optimizers = torch.optim.SGD(cnn.parameters(),lr=0.001,momentum=0.9)
losses = torch.nn.MSELoss()
cnn.train()

for i in range(25):
    acc = 0

    for img,label in data:

        lb = OneHotEncoder(categories='auto')
        lb.fit(label.reshape(-1,1))
        label = lb.transform(label.reshape(-1,1)).toarray()

        img = Variable(img)
        label = torch.from_numpy(label).float()
        label = Variable(label)
        optimizers.zero_grad()
        predict = cnn(img)

        loss = losses(predict,label)

        loss.backward()

        optimizers.step()

        a = torch.argmax(predict,dim=1)

        b = torch.argmax(label.data,dim=1)

        for k in range(len(a)):
            if(a[k] == b[k]): acc += 1
        print(loss)
    print("accuracy:%2f"%(acc/4128))
torch.save(cnn.state_dict(),'sex.pth')
