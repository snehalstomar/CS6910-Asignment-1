import numpy as np
import myDLkit2

number_hidden_layers  = 2
num_classes =10 #depends on output
num_inputs=784
number_neurons = [num_inputs, 64, 32, num_classes]
activation = "tanh"
initialization = "xavier"
test_input = np.random.randint(0, 256, (784, 1))



nn = myDLkit2.feed_fwd_nn(number_neurons, activation, number_hidden_layers, initialization)
nn.back_prop(test_input, 'cross-entropy', 9)
print("error")
print(nn.error)
print("accuracy")
print(nn.accuracy)
print("========grad_A================")
print(nn.grad_A)
print("========grad_H================")
print(nn.grad_H)
print("========grad_W================")
print(nn.grad_Weights)
print("========grad_B================")
print(nn.grad_Biases)
#print(len(nn.Weights))
	