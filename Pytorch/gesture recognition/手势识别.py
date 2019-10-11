# %% [code]
import torch
import torch.optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset,DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

train = pd.read_csv('../input/sign-language-mnist/sign_mnist_train.csv',dtype=np.float32)


y = train.label.values
x = train.loc[:,train.columns!= 'label'].values / 255


train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.2,random_state=42)
print(test_x.shape)
train_x = torch.from_numpy(train_x)
train_y = torch.from_numpy(train_y).type(torch.LongTensor)

test_x = torch.from_numpy(test_x)
test_y = torch.from_numpy(test_y).type(torch.LongTensor)


train = TensorDataset(train_x,train_y)
test = TensorDataset(test_x,test_y)

train_loader = DataLoader(train,batch_size=100,shuffle=False,drop_last=True)
test_loader = DataLoader(test,batch_size=100,shuffle=False,drop_last=True)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(1,16,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(16,32,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layers4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

        )
        self.fc = nn.Sequential(
            nn.Linear(7*7*128,1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024,100),
            nn.ReLU(inplace=True),
            nn.Linear(100,26)
        )
    def forward(self, x):
        x = self.layers1(x)
        x = self.layers2(x)
        x = self.layers3(x)
        x = self.layers4(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)

        return x




cnn = CNN()

error = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(cnn.parameters(),lr=0.1)

for epochs in range(100):
    for i ,(img,label) in enumerate(train_loader):
        img = img.view(100,1,28,28)
        img = Variable(img)
        label = Variable(label)
        optimizer.zero_grad()
        output = cnn(img)

        loss = error(output,label)
        loss.backward()

        optimizer.step()

        if i %50 == 0:
            accuracy = 0
            for x,y in test_loader:

                x = x.view(100,1,28,28)
                x = Variable(x)


                out = cnn(x)
                pre = torch.max(out.data,1)[1]
                accuracy += (pre == y).sum()

            print('accuracy:',accuracy.item()/(5491))
            if(accuracy.item()/(5491)>0.975):
                torch.save(cnn.state_dict(),'../working/abcd.pth')
            if(accuracy.item()/5491 > 0.983):
                break