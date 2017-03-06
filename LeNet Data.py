#1 
assert(len(X_train))==len(y_train)
#使用assert断言是学习python一个非常好的习惯，python assert 断言句语格式及用法很简单。在没完善一个程序之前，我们不知道程序在哪里会出错，与其让它在运行最崩溃，不如在出现错误条件时就崩溃，这时候就需要assert断言的帮助。本文主要是讲assert断言的基础知识。
#python assert断言是声明其布尔值必须为真的判定，如果发生异常就说明表达示为假。可以理解assert断言语句为raise-if-not，用来测试表示式，其返回值为假，就会触发异常。
#2
print("training Set: {} samples".format(len(X_train)))
#3
X_train = np.pad(X_train,((0,0),(2,2),(2,2),(0,0)),'constant')
#http://blog.csdn.net/liyaohhh/article/details/51111115,no para to constant means put 0 in it
print("shape is :{}".format(X_train[0].shape)
#4
import random
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
index= random.randint(0,len(X_train))
image = X_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image,cmap="gray")
#http://stackoverflow.com/questions/25453587/unwanted-extra-dimensions-in-numpy-array,when X_train[index],the first dimension is [[]],we need to remove


#5
from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

#tensor flow part
#1
import tensorflow as tf 
EPOCHS=10 #how many times we train our data 
BATCH_SIZE = 128 # how many train images we run at a time 

from tensorflow.contrib.layers import flatten 
def LeNet(x):
  #setting hyper parameters
  mu = 0
  sigma = 0.1
  #
