'''
CS6910-Assignment-1
Submitted by:
1.EE20S006 Snehal Singh Tomar
2. EE20D006 Ashish Kumar
====================================================
Tarining the NN using SGD Assignment-1-Q-3-1
'''
#Importing Libraris and Self-Defined Headers
import numpy as np
import myDLkit2
import myDataExtractor

#Hyper-parmeters
input_data = myDataExtractor.data
number_hidden_layers  = 2
num_classes =10 #depends on output
activation = "sigmoid"
initialization = "xavier"
num_inputs=input_data[0][0].shape[0]
number_neurons = [num_inputs, 64, 32, num_classes]
training_specifics = ["sgd", 5, 100, 1*np.exp(-4), 0, 0, 0, "cross-entropy", input_data] 

#Instantitation(Creation of Neural Network)
nn = myDLkit2.feed_fwd_nn(number_neurons, activation, training_specifics, number_hidden_layers, initialization)
print("initial Biases==>", nn.Biases)
#Training
nn.train()

print("Trained Biases==>",nn.Biases)
