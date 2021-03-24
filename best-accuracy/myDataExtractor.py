# -*- coding: utf-8 -*-
"""Untitled8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1oboX_LaXJGq9oLwWaQ8cMS4KxbrobFks
"""

import numpy as np
from tensorflow.keras.datasets import fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

def data_processing(x,y):
  x = x.astype(np.float)
  #normalization
  data_norm = x/255.0 
  # data shuffle
  index = np.random.permutation(data_norm.shape[0])
  x = data_norm[index]
  y = y[index]

  #one hot encoding (labels)

  y_vec = np.zeros((len(y),1,10))
  for i in range(len(y)):
    y_vec[i,0,y[i]] = 1
  
  return x,y_vec,y

def flatten(x):
  data = np.zeros((x.shape[0],1,x.shape[1]*x.shape[2]))
  for i in range(x.shape[0]):
    data[i,0,:] = x[i].reshape(1,-1)
  return data

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train,y_train_onehot,y_train = data_processing(x_train,y_train)
x_test,y_test_onehot,y_test = data_processing(x_test,y_test)


x_train = flatten(x_train)
x_test = flatten(x_test)


M =x_train.shape[0]
M = int(M*0.1)       #10%  data as validation



x_valid = x_train[0:M,:,:]        # x_train and y_train is already shuffled hence directly sampling validation sets
y_valid_onehot = y_train_onehot[0:M,:,:]
y_valid = y_train[0:M]


x_train = x_train[M:,:,:]
y_train = y_train[M:]
y_train_onehot = y_train_onehot[M:,:,:]

#print(x_train.shape)
#print(y_train.shape)
#print(y_train_onehot.shape)


#print(x_valid.shape)
#print(y_valid.shape)
#print(y_valid_onehot.shape)

train =[]
for i in range(x_train.shape[0]):
  x = x_train[i,0,:]
  y = y_train[i]           #replace by y =y_train_Onehot[i,0,:] for one hot
  tple = (x.reshape(784,1),y)
  train.append(tple)

valid =[]
for i in range(x_valid.shape[0]):
  x = x_valid[i,0,:]
  y = y_valid[i]           #replace by y =y_train_Onehot[i,0,:] for one hot
  tple = (x.reshape(784,1),y)
  valid.append(tple)

#x_test.shape

test =[]
for i in range(x_test.shape[0]):
  x = x_test[i,0,:]
  y = y_test[i]
  tpl = (x.reshape(784,1),y)
  test.append(tpl)





