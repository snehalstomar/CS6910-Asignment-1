'''
CS6910-Assignment-1
Submitted by:
1.EE20S006 Snehal Singh Tomar
2. EE20D006 Ashish Kumar
====================================================
Tarining the NN using SGD Assignment-1-Q-3-1
'''
#Importing Libraries and Self-Defined Headers
import numpy as np
import myDLkit2
import myDataExtractor

import wandb

hyperparameter_defaults = dict(
    activation = "sigmoid",
    initialization = "xavier",
    eta = 0.001,
    n_epochs = 2,
    neurons_layer_1 = 64,
    neurons_layer_2 = 32,
    )

wandb.init(config=hyperparameter_defaults, project="cs6910-assignment-1")
config = wandb.config

#Hyper-parmeters
input_data = myDataExtractor.data
number_hidden_layers  = 2
num_classes =10 #depends on output
activation = config.activation
initialization = config.initialization
num_inputs=input_data[0][0].shape[0]
number_neurons = [num_inputs, config.neurons_layer_1, config.neurons_layer_2, num_classes]
training_specifics = ["mbgd", config.n_epochs, 100, config.eta, 0.1, 0, 0, "cross-entropy", input_data] #optimizer,epochs,batchsz,eta, gamma, beta1 &@, loss, data

#Instantitation(Creation of Neural Network)
nn = myDLkit2.feed_fwd_nn(number_neurons, activation, training_specifics, number_hidden_layers, initialization)
#print("initial Biases==>", nn.Biases)
#print("initial Weights==>", nn.Weights)
#Training
nn.train()
accuracy = nn.accuracy
metrics = {'accuracy': accuracy}
wandb.log(metrics)
#print("Trained Biases==>",nn.Biases) 	
#print("Trained Weights==>", nn.Weights)
#print(nn.accuracy)