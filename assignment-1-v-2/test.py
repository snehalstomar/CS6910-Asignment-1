import numpy as np
import myDLkit2

number_hidden_layers  = 2
num_classes =10 #depends on output
number_neurons = [784, 64, 32, num_classes]
activation = "tanh"
initialization = "xavier"
test_input = np.random.randint(0, 256, (784, 1))



nn = myDLkit2.feed_fwd_nn(number_neurons, activation, number_hidden_layers, initialization)
y_hat = nn.fwd_prop(test_input)
print(y_hat)
