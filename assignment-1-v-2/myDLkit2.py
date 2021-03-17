import numpy as np

#Defining all activation functions and their derivatives one by one
def acivation_relu(X):
    R = np.maximum(0, X)
    return R

def activation_sigmoid(X):
    S = 1 / (1 + np.exp(-1 * X))
    return S

def activation_tanh(X):
	T=np.tanh(X)
	return T

def derivative_relu(X):
    X[X >= 0] = 1
    X[X < 0]  = 0
    return X

def derivative_sigmoid(X):
    SD = sigmoid(X) * (1 - sigmoid(X))
    return SD

def derivative_tanh(X):
	TD = 1 - np.square(activation_tanh(X))
	return TD

def softmax(X):
	e_pow_x = np.exp(X)
	return e_pow_x / e_pow_x.sum()


#Defining the Neural Network Class

class feed_fwd_nn:
	
	#Definition of Constructor
	#number_layers = number of hidden layers 
	#number_neurons_in_layers = list wherein the ith element represents the number of neurons in the ith layer. 
		#0th element->number of elements in input feture vector , last element-> number of bins in which the data needs to be classified  
	#initialization = random/xavier
	#number_outputs = 
	#activation = Str indicating the type of activation to be used
	def __init__(self, number_neurons_in_layers, activation, number_layers: int = 1, initialization: str= "random"):
		
		#Defining preliminaries
		self.num_layers = number_layers+1 #to accomodate input layer as well
		self.neurons = number_neurons_in_layers #its length should be equal to self.num_layers+1
		self.init = initialization
		self.activation = activation
		self.Weights = [] #list containing weights corresponding to each layer
		self.Biases = [] #list containing biases corresponding to each layer
		
		self.A = [] #list to store column vector representing pre-activation value of ith layer
		self.H = []	#list to store column vector representing activation value of ith layer
		
		self.A_output_layer = np.zeros([self.neurons[-1], 1]) #column vector for storing output layer's activation values
		self.y_hat = np.zeros([self.neurons[-1], 1])

		for i in range(self.num_layers):
			self.A.append(np.zeros([self.neurons[i], 1]))
			self.H.append(np.zeros([self.neurons[i], 1]))

		#Initialising a n_ouput_layerXn_input_layer Weight matrix
		#Initialising a n_ouput_layerX1  Bias col vector
		if self.init == "random":
			for i in range(self.num_layers):
				self.Weights.append(np.random.randn(self.neurons[i+1], self.neurons[i]))
				self.Biases.append(np.random.randn(self.neurons[i+1], 1))#.reshape(neurons[i+1], 1)|
		elif self.init == "xavier":
			for i in range(self.num_layers):
				self.Weights.append(np.random.randn(self.neurons[i+1], self.neurons[i])*np.sqrt(1/(self.neurons[i+1] + self.neurons[i+1])))
				self.Biases.append(np.random.randn(self.neurons[i+1], 1) * np.sqrt(1/(self.neurons[i+1] + self.neurons[i+1])))#.reshape(neurons[i+1], 1)|
		else:
			raise Exception("Invalid initialization method")

	#Definition of the forward propagation function
	#input_feature_vector ~ col vector representing input		
	def fwd_prop(self, input_feature_vector):
		for i in range(self.num_layers):
			if i == 0:
				self.A[i] = input_feature_vector
				self.H[i] = self.A[i]
			else:
				self.A[i] = np.dot(self.Weights[i-1], self.H[i-1]) + self.Biases[i-1]

				if self.activation == "relu":
					self.H[i] = acivation_relu(self.A[i])
				elif self.activation == "sigmoid":
					self.H[i] = acivation_sigmoid(self.A[i])
				elif self.activation == "tanh":
					self.H[i] = activation_tanh(self.A[i])
				else:
					raise Exception("Invalid activation method")					 
		self.A_output_layer = np.dot(self.Weights[-1], self.A[-1]) + self.Biases[-1]
		self.y_hat = softmax(self.A_output_layer)
		return self.y_hat


