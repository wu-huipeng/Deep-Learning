### **Faster RCNN 网络实现步骤（不包括损失函数部分）**
---

**一 第一步**
 - 将图片输入的VGG16网络中，得到feature-map
 - 设置框的三种比例和三种尺寸，一个有九个框，遍历feature-map的每一个像素点，以其为中心生成anchor
 - 根据原始的图片尺寸，将anchor中没有超出尺寸大小的anchor标记出来
 
**二 第二步**
 - 计算每一个anchor和真实框的IOU
 - 然后根据IOU的大小，将anchor分成正负样本和无关样本
 
