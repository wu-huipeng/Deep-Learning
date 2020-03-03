
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2

#标签
INDEX = {"holothurian":0,"echinus":1,"scallop":2,"starfish":3}

#获取box和image的路径
def get_path():
    path = os.listdir("./")
    for name in path:
        if(name[-2:] == "py"):
            path.remove(name)

    img_path = os.listdir(os.path.join("./",path[1]))
    box_path = os.listdir(os.path.join("./",path[0]))

    img_path = [os.path.join("./",path[1],i) for i in img_path]
    box_path = [os.path.join("./",path[0],i) for i in box_path]
    return img_path,box_path


#读取图片和标注值
def read_image_box(img_path,box_path):
    image = cv2.imread(img_path)
    label = []
    real_bbox = []
    xml_tree = ET.parse(box_path)

    root = xml_tree.getroot()

    for object in root.findall("object"):
        if object.find("name").text == "waterweeds":  #忽略这一类
            continue
        label.append(INDEX[object.find("name").text])

        for bndbox in object.findall("bndbox"):
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            real_bbox.append(np.array([xmin,ymin,xmax,ymax]))

    return image,np.array(label),np.array(real_bbox)


#加载训练集
class DataSet(Dataset):
    def __init__(self):
        self.image_path,self.box_path = get_path()


    def __getitem__(self, index):
        img_path = self.image_path[index]
        box_path = self.box_path[index]
        img,label,bbox = read_image_box(img_path,box_path)

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        bbox = torch.from_numpy(bbox)

        return img,label,bbox


    def __len__(self):
        return len(self.image_path)
