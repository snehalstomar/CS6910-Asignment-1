'''
CS6910-Assignment-1-Q3.i-Stochastic Gradient Descent
Submitted by:
1. EE20S006 Snehal Singh Tomar
2. EE20D006 Ashish Kumar
'''
#Importing necessary libraries 
import numpy as np
import myDLkit#(Our Principal Header-Self Defined)
#import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#defining variables and pre-requisites
n_inputs = 784
n_hidden_layers = 5
n_outputs = 10
eta = 0.01
gamma = 0.9
errors_train = []
errors_test = []
train_error_post_epoch = []
accuracy = []

#normalizing image data and converting it to feature vectors
def data_preprocessing(features):
  '''
  features: N * w * h array   (images)
  this function normalizes the intensity values
  '''
  data = features.astype(np.float)        #convert to float
  data_norm = data/255.0                 # normalized to 0 and 1
  return data_norm

#Instantiating(Creating the neural Network)
nn = myDLkit.feed_fwd_nn(n_hidden_layers, n_inputs, n_outputs)

#Training on the Train-Dataset(Using Stochastic Gradient Descent, Cross Entropy Loss)
epochs_train = 2 #number of epochs
err = 0       #track error after each iteration/epochs
print("training")
for i in range(epochs_train):       # iterations = n(epochs)
  for j in range(x_train.shape[0]):         #itetrate over all examples
    print("processing image %d of epoch %d" %(j, i))
    norm_x = data_preprocessing(x_train)
    nn.backProp((norm_x[j,:,:]).reshape(1,-1),y_train[j])      #pass one example at a time
    errors_train.append(nn.error)
    err = err + nn.error
    for m in nn.grad_wrt_b:
      nn.layers[m].b = nn.layers[m].b - (eta*nn.grad_wrt_b[m].T)
    for p in nn.grad_wrt_W:
      nn.layers[p].W = nn.layers[p].W - (eta*nn.grad_wrt_W[p])
  print("=============================") #Reporting error and accuracy post each epoch
  print("The overall error after %s epoch: %s" %(i,err/x_train.shape[0]))
  
  train_error_post_epoch.append(err/x_train.shape[0])
  accuracy.append(nn.accuracy_counter)
  nn.accuracy_counter = 0

#Testing on the Test-Dataset
err = 0       #track error after each iteration/epochs
#for i in range(1):       # iterations = n(epochs)
print("testing")
for j in range(x_test.shape[0]):         #itetrate over all examples
  print("testing wrt %d test image" %(j))
  norm_x = data_preprocessing(x_test)
  nn.backProp((norm_x[j,:,:]).reshape(1,-1),y_test[j])      #pass one example at a time
  errors_test.append(nn.error)
  err = err + nn.error
  
print("=============================")
print("The overall test-error is  %s. and overall accuracy is %s" %(err/x_test[0], nn.accuracy))

#Visualizing The results
plt.subplot(3, 1, 1)
x = np.arange(0, x_train.shape[0]*epochs_train, 1)
plt.title("Training errors over each example")
plt.plot(x, errors_train[x])

plt.subplot(3, 1, 2)
y = np.arange(0, y_test.shape[0], 1)
plt.title("Test errors over each example")
plt.plot(y, errors_test[y])

plt.subplot(3, 1, 3)
z = np.arange(0, epochs_train, 1)
plt.title("accuracy over training epochs")
plt.plot(z, errors_test[z])

plt.show()