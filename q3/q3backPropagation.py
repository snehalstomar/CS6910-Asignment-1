import numpy as np
import myDLkit

n_inputs = 10
n_hidden_layers = 8
n_outputs = 10
in_row_vector = np.zeros([1, 10])
for i in range(n_inputs):
	in_row_vector[0, i] = i

nn = myDLkit.feed_fwd_nn(n_hidden_layers, n_inputs, n_outputs)
grad_wrt_h ={} #key i of grad_wrt_h correponds to layer i+1 in layers[] of my DL kit {Dictionary with number of keys = number of hidden layers }
grad_wrt_a ={} #dDictionary with number of keys = number of hidden layers +1 ; last for a's of output layer }
grad_wrt_b = {}
grad_wrt_W = {}
def backProp(neural_net_obj, input_vector, true_label):
	y_hat = neural_net_obj.forwardProp(input_vector)
	y_actual = np.zeros([1, 10])
	y_actual[0, true_label] = 1
	#gradient with respect to output layer for cross entropy loss
	grad_vec_wrt_output = -(y_actual.T - y_hat.T);#(kx1)
	grad_wrt_a[neural_net_obj.n_hidden_layers+1] = grad_vec_wrt_output
	#print(grad_vec_wrt_output)
	#gradient with respect to 'h' last hidden layer
	h_grad_last = np.zeros([neural_net_obj.n_inputs, 1])
	a_grad_last = np.zeros([neural_net_obj.n_inputs, 1])
	for i in range(neural_net_obj.n_inputs):
		h_grad_last[i, 0] = np.dot(neural_net_obj.last_hidden_layer_weights[:,i].T, grad_vec_wrt_output)
	grad_wrt_h[neural_net_obj.n_hidden_layers] = h_grad_last
	
	for i in range(neural_net_obj.n_inputs):
		a_grad_last[i, 0] = h_grad_last[i,0] * neural_net_obj.last_hidden_layer_h[0, i] * (1 - neural_net_obj.last_hidden_layer_h[0, i])
	grad_wrt_a[neural_net_obj.n_hidden_layers] = a_grad_last	  	
	grad_wrt_b[neural_net_obj.n_hidden_layers] = grad_wrt_a[neural_net_obj.n_hidden_layers]
	grad_wrt_W[neural_net_obj.n_hidden_layers] = np.dot(grad_wrt_a[neural_net_obj.n_hidden_layers+1],neural_net_obj.last_hidden_layer_h)
	
	for i in range(neural_net_obj.n_hidden_layers-1, 0, -1):
		grad_wrt_h[i] = np.dot(neural_net_obj.layers[i].W.T, grad_wrt_a[i+1])	
		grad_a_i = np.zeros([neural_net_obj.n_inputs, 1])
		for j in range(neural_net_obj.n_inputs):
			grad_a_i[j, 0] = grad_wrt_h[i][j,0] * neural_net_obj.layers[i].h[0,j] * (1 - neural_net_obj.layers[i].h[0,j]) 
		grad_wrt_a[i] = grad_a_i
		grad_wrt_b[i] = grad_wrt_a[i] 
		grad_wrt_W[i] = np.dot(grad_wrt_a[i+1], grad_wrt_h[i].T) 
	grad_wrt_W[0] = np.dot(grad_wrt_a[1], (np.dot(neural_net_obj.layers[0].W.T, grad_wrt_a[1])).T)	
		


backProp(nn, in_row_vector, 2)	
print("grad_wrt_h")
print(grad_wrt_h)
print("grad_wrt_a")
print(grad_wrt_a)

print("grad_wrt_W")
print(grad_wrt_W)

print("grad_wrt_b")
print(grad_wrt_b)