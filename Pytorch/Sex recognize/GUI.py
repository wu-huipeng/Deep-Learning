from tkinter import *
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T
import tkinter.filedialog


def sex():

    class CNN(torch.nn.Module):
        def __init__(self):
            super(CNN, self).__init__()

            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 64
            )

            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 32
            )

            self.layer3 = torch.nn.Sequential(
                torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU(inplace=True)
            )
            self.layer4 = torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(kernel_size=2, stride=2)  # 16
            )
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(128 * 16 * 16, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, 2048),
                torch.nn.ReLU(inplace=True),
                torch.nn.Linear(2048, 2)
            )

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = x.view(x.size(0), -1)
            x = F.softmax(self.fc(x), dim=1)
            return x
    cnn = CNN()

    cnn.load_state_dict(torch.load('sex.pth'))
    img = Image.open(file_path)
    ts = T.Compose([
        T.Resize(128),
        T.CenterCrop(128),
        T.ToTensor(),
        T.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    ])
    img = ts(img)
    img = torch.unsqueeze(img,0)
    predict = cnn(img)
    b =torch.argmax(predict,dim=1)
    text1.delete(1.0, "end")
    if(b==0):text1.insert(INSERT,"女")
    else:text1.insert(INSERT,"男")


def get_root():
    global file_path
    root = Tk()
    root.withdraw()

    file_path = tkinter.filedialog.askopenfilename()


def main():

    global root,text1
    root = Tk()
    root.geometry("450x300")
    root.title("性别识别")


    Label(root,text="预测结果:",font=("微软雅黑",15),fg="black").place(x=100,y=260)

    text1 = Text(root,width=5,height=2)
    text1.place(x=200,y=260)
    Button(root, text="打开图片", width=10, height=2, command=get_root, fg='black', bg='gray').place(x=150, y=70)

    Button(root,text="确定",width=10,height=2,command=sex,fg='red',bg='gray').place(x=200,y=200)


    root.mainloop()

if __name__ == '__main__':
    main()

