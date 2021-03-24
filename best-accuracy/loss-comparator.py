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
    loss = "squared-error",
    eta = 0.01
    )

wandb.init(config=hyperparameter_defaults, project="cs6910-assignment-1")
config = wandb.config
#Hyper-parmeters
input_data = myDataExtractor.valid #validation data set
number_hidden_layers  = 3
num_classes =10 #depends on output
activation = "tanh"
initialization = "xavier"
num_inputs=input_data[0][0].shape[0]
if number_hidden_layers == 3:
	number_neurons = [num_inputs, 64, 32, 32, num_classes]
elif number_hidden_layers == 4:
	number_neurons = [num_inputs, 64, 32, 32, 32, num_classes]
elif number_hidden_layers == 5:
	number_neurons = [num_inputs, 64, 32, 32, 32, 32, num_classes]

training_specifics = ["sgd", 5, 16, config.eta, 0.1, 0.1, 0.1, config.loss, input_data] #optimizer,epochs,batchsz,eta, gamma, beta1 &@, loss, data
        
#Instantitation(Creation of Neural Network)
nn = myDLkit2.feed_fwd_nn(number_neurons, activation, training_specifics, number_hidden_layers, initialization)

#wandb.watch(nn)
#print("initial Biases==>", nn.Biases)
#print("initial Weights==>", nn.Weights)
#Training
nn.train()
accuracy = nn.accuracy
loss = nn.lossval
metrics = {'accuracy': accuracy, 'loss':loss}
wandb.log(metrics)

#print("Trained Biases==>",nn.Biases) 	
#print("Trained Weights==>", nn.Weights)
#print(nn.accuracy)