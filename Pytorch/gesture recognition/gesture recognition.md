# 基于pytorch的手势识别
---

一. 项目简要说明
    
   1. 该项目是识别26(a-z)个英文字母的手势，难度不大。
   2. 数据集可以在kaggle上面下载，数据的格式是csv格式的数据集，和我们的MNIST数据差不多。
   3. 训练好了的模型和数据集可以在我的kaggle上面下载[**地址**](https://www.kaggle.com/huipengwu/kernel6259e4e878).
   

二. 网络和优化及损失函数说明
    
   1. 我们选用的网络是AlexNet网络。
   2. 优化函数是随机梯度下降SGD，学习率为0.1。
   3. 损失函数是交叉熵损失函数CrossEntropyLoss。
 
三. 总结

   1. 在构造数据的时候需要用类似于TensorFlow方式，而不能重写Dataset，一开始，我重写Dataset训练的时候，准确率一直只有0.07%左右的准确率，
   用这个TensorDataset(train_x,train_y)，后然后再使用DataLoader，即可
   2. 其他的步骤和之前的项目的步骤几乎相同，照葫芦画瓢即可。
