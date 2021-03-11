'''
 Python Program for CS6910-Assignment-1-Q-3:Demonstration of working of Back Propagation
 EE20S006-Snehal Singh Tomar
 EE20D006-Ashish Kumar
'''

import numpy as np

import myDLkit #importing our self-defined header containing the feed forward neural network class
#loading the dataset
import numpy as np
import myDLkit
from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#definiing hyper parameters of the nn to be created
n_inputs = 784
n_hidden_layers = 20
n_outputs = 10


#defining a test input 
img1 = x_train[0]
label = y_train[0]
def img_normalized_feature_extractor(img):
	img_arr = np.zeros([1, img.shape[0]*img.shape[1]])
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img_arr[0, (img.shape[1]*i)+j] = img[i, j] / 255 #normalizing wrt intensities
	return img_arr
img_one_d = img_normalized_feature_extractor(img1) #the feature vector

#creating the neural network. 'nn' is the neural network object
nn = myDLkit.feed_fwd_nn(n_hidden_layers, n_inputs, n_outputs)

#calling  the backpropagation function on the neural net object
nn.backProp(img_one_d, label) #nn_object.backProp(input_vector, true label(class number) to which the feature vector(image) belongs)	

#assuming eta = 1 for SGD
#updates for other learning algos can also be done on similar lines
#updating the biases post calculation of grads by backprop
for i in nn.grad_wrt_b:
	nn.layers[i].b = nn.layers[i].b - nn.grad_wrt_b[i].T	

#updating the weights post calculation of grads by backprop
for i in nn.grad_wrt_W:
	nn.layers[i].W = nn.layers[i].W - nn.grad_wrt_W[i]


#This section can be uncommented to view the results.	
#printing the updated weights and biases
for i in nn.grad_wrt_b:
	print(nn.layers[i].b)

for i in nn.grad_wrt_W:
	print(nn.layers[i].W) 


