'''
CS6910-Assignment-1
Submitted by:
1.EE20S006 Snehal Singh Tomar
2. EE20D006 Ashish Kumar
=========================================
Extraction of Data and its-preprocessing
'''
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

x_train,y_train_onehot,y = data_processing(x_train,y_train)


x_train = flatten(x_train)

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

data =[]
for i in range(x_train.shape[0]):
  x = x_train[i,0,:]
  y = y_train[i]           #replace by y =y_train_Onehot[i,0,:] for one hot
  tple = (x.reshape(784,1),y)
  data.append(tple)

