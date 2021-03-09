import numpy as np

class feed_fwd_nn:

	class hidden_layer:
		
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
			
		def activation(self):
			for i in range(self.size):
				self.h[0, i] = 1 / (1 + np.exp(self.a[0, i]))

	class input_layer:

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

	def forwardProp(self, data_inputs):
		print("In input Layer")
		print("self.n_hidden_layers = ", self.n_hidden_layers)
		self.layers[0].a = data_inputs
		self.layers[0].activation()
		for i in range(1, self.n_hidden_layers):
			print("======================================")
			print("processing hidden_layer: %d" % (i))
			self.layers[i].a = (np.dot(self.layers[i-1].W, self.layers[i-1].h.T) + self.layers[i-1].b.T).T 
			print("hidden layer(", i, ").a=", self.layers[i].a)
			self.layers[i].activation()
			print("hidden layer(", i, ").h=", self.layers[i].h)
		self.layers.append(self.layer_before_output(self.n_inputs, self.n_outputs, self.layers[self.n_hidden_layers-1].h))												
		print("entering last hidden layer")
		self.layers[self.n_hidden_layers].activation_sigmoid()
		print("processed last hidden layer")
		self.out_layer.a = (np.dot(self.layers[self.n_hidden_layers-1].W, self.layers[self.n_hidden_layers-1].h.T) + self.layers[self.n_hidden_layers-1].b.T).T
		print("processing output function")	
		self.out_layer.output_fn()
		return self.out_layer.h


x = np.zeros([1, 20])
for i in range(20):
	x[0, i] = i

nn = feed_fwd_nn(7, 20, 10)
y = nn.forwardProp(x)
print(y)
		
		






