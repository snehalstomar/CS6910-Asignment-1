'''
 Python Program for CS6910-Assignment-1-Q-2
 EE20S006-Snehal Singh Tomar
 EE20D006-Ashish Kumar
'''
#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#defining the feed_fwd_nn class; the nn will be an instance of this class
class feed_fwd_nn:

	class hidden_layer:
		#each hidden layer will be an instance of this class
		def __init__(self, size: int = 0, data_inputs: np.ndarray = np.zeros([1, 784])):
			self.a = np.zeros([1, size])
			self.b = np.zeros([1, size])
			self.h = np.zeros([1, size])
			self.a = data_inputs
			self.W = np.zeros([size, size])
			self.size = size
			for i in range(size):
				for j in range(size):
					self.W[i, j] = 1.0 
		#The sigmoid activation function	
		def activation(self):
			for i in range(self.size):
				self.h[0, i] = 1 / (1 + np.exp(self.a[0, i]))

	class input_layer:
		#the input layer will be an instance of this class, it will just pass on the input vector, as it were
		def __init__(self, size: int, data_inputs: np.ndarray = np.zeros([1, 784])):
			self.a = np.zeros([1, size])
			self.b = np.zeros([1, size])
			self.h = np.zeros([1, size])
			self.a = data_inputs
			self.W = np.zeros([size, size])
			self.size = size
			for i in range(size):
				for j in range(size):
					self.W[i, j] = 1.0

		def activation(self):
			self.h[:] = self.a[:]
		 
			
	class layer_before_output:
		#the last hidden layer class. this will be asymmetric ie. having 'n' inputs and 'k' outputs
		def __init__(self, size_in: int, size_out: int, data_inputs: np.ndarray = np.zeros([1, 784])):
			self.a = np.zeros([1, size_in])
			self.b = np.zeros([1, size_out])
			self.a = data_inputs
			self.h = np.zeros([1, size_in])
			self.W = np.zeros([size_out, size_in])
			self.size_in = size_in
			self.size_out = size_out 

		def activation_sigmoid(self):
			for i in range(self.size_in):
				self.h[0, i] = 1 / (1 + np.exp(self.a[0, i]))
		
	class output_layer:
		#the output layer, we chose a softmax output function, since the objective is classification
		def __init__(self, size: int, data_inputs: np.ndarray = np.zeros([1, 784])):
			self.a = np.zeros([1, size])
			self.h = np.zeros([1, size])
			self.a = data_inputs
			self.size = size
			self.avg = 0
		def averager(self):
			for i in range(self.size):
				self.avg += np.exp(self.a[0, i])
		#softmax output		
		def output_fn(self):
			self.averager()
			for i in range(self.size):
				self.h[0, i] = np.exp(self.a[0, i]) / self.avg 			

	#constructor for the neural network class object
	def __init__(self, n_hidden_layers: int = 0, n_input_vector: int = 0, n_output_vector: int = 0):
		self.n_inputs = n_input_vector
		self.n_outputs = n_output_vector
		self.n_hidden_layers = n_hidden_layers
		self.layers = []
		self.layers.append(self.input_layer(self.n_inputs))
		for i in range(self.n_hidden_layers-1):
			self.layers.append(self.hidden_layer(self.n_inputs))
		self.l_Lminus1 = self.layer_before_output(self.n_inputs, self.n_outputs)
		self.out_layer= self.output_layer(self.n_outputs)
	#the forward propagation function to generate all a, h and y_hat for a given x	
	def forwardProp(self, data_inputs):
		print("In input Layer")
		print("self.n_hidden_layers = ", self.n_hidden_layers)
		self.layers[0].a = data_inputs
		self.layers[0].activation()
		for i in range(1, self.n_hidden_layers):
			print("======================================")
			print("processing hidden_layer: %d" % (i))
			#pre-activation of hidden layers
			self.layers[i].a = (np.dot(self.layers[i-1].W, self.layers[i-1].h.T) + self.layers[i-1].b.T).T 
			#print("hidden layer(", i, ").a=", self.layers[i].a)
			#activation of hidden layers
			self.layers[i].activation()
			#print("hidden layer(", i, ").h=", self.layers[i].h)
		self.layers.append(self.layer_before_output(self.n_inputs, self.n_outputs, self.layers[self.n_hidden_layers-1].h))												
		print("entering last hidden layer")
		self.layers[self.n_hidden_layers].activation_sigmoid()
		print("processed last hidden layer")
		self.out_layer.a = (np.dot(self.layers[self.n_hidden_layers].W, self.layers[self.n_hidden_layers].h.T) + self.layers[self.n_hidden_layers].b.T).T
		print("processing output function")	
		self.out_layer.output_fn()
		return self.out_layer.h
#-------------------------------------------------------------------
#reading the first image from training datatset as an np array and converting it to a row(feature) vector
img1 = x_train[0]
def img_normalized_feature_extractor(img):
  img_arr = np.zeros([1, img.shape[0]*img.shape[1]])
  for i in range(img.shape[0]):
    for j in range(img.shape[1]):
      img_arr[0, (img.shape[1]*i)+j] = img[i, j] / 255 #normalizing wrt intensities
      return img_arr
img_one_d = img_normalized_feature_extractor(img1) #the feature vector

#instantiating the nn class by defining the number of hidden layers, dimension of feature vector and dimension of y_hat
nn = feed_fwd_nn(10, img1.shape[0]*img1.shape[1], 10)
#calling the forwardProp function on the nn object
y_hat = nn.forwardProp(img_one_d)
#printing y_hat (ith element deonotes the output proability of x being assigned to ith class)
print(y_hat)







