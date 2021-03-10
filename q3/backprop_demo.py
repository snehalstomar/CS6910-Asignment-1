'''
 Python Program for CS6910-Assignment-1-Q-3:Demonstration of working of Back Propagation
 EE20S006-Snehal Singh Tomar
 EE20D006-Ashish Kumar
'''
import numpy as np
import myDLkit #importing our self-defined header containing the feed forward neural network class

#definiing hyper parameters of the nn to be created
n_inputs = 10
n_hidden_layers = 8
n_outputs = 10

#defining a test input 
in_row_vector = np.zeros([1, 10])
for i in range(n_inputs):
	in_row_vector[0, i] = i

#creating the neural network. 'nn' is the neural network object
nn = myDLkit.feed_fwd_nn(n_hidden_layers, n_inputs, n_outputs)

#calling  the backpropagation function on the neural net object
nn.backProp(in_row_vector, 2) #nn_object.backProp(input_vector, true label(class number) to which the feature vector(image) belongs)	

#updating the biases post calculation of grads by backprop
for i in nn.grad_wrt_b:
	nn.layers[i].b = nn.layers[i].b + nn.grad_wrt_b[i].T	

#updating the weights post calculation of grads by backprop
for i in nn.grad_wrt_W:
	nn.layers[i].W = nn.layers[i].W + nn.grad_wrt_W[i].T

#printing the updated weights and biases
for i in nn.grad_wrt_b:
	print(nn.layers[i].b)

for i in nn.grad_wrt_W:
	print(nn.layers[i].W)


		