'''
CS6910-Assignment-1
Submitted by:
1.EE20S006 Snehal Singh Tomar
2. EE20D006 Ashish Kumar
====================================================

'''
#Importing Libraries and Self-Defined Headers
import numpy as np
import myDLkit2
import myDataExtractor
import wandb

hyperparameter_defaults = dict(
    optimizer = 'sgd',
    batchsz = 16,
    init = 'xavier',
    activation = 'tanh',
    num_hidden_layers = 3,
    learning_rate = 0.001,
    epochs = 10,
    )
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress','Coat', 'Sandal',   'Shirt','Sneaker', 'Bag' , 'Ankle boot'] 

wandb.init(config=hyperparameter_defaults, project="cs6910-assignment-1")
config = wandb.config
#Hyper-parmeters
input_data = myDataExtractor.test #test data set
number_hidden_layers  = config.num_hidden_layers
num_classes =10 #depends on output
activation = config.activation
initialization = config.init
num_inputs=input_data[0][0].shape[0]
if number_hidden_layers == 3:
	number_neurons = [num_inputs, 64, 32, 32, num_classes]
elif number_hidden_layers == 4:
	number_neurons = [num_inputs, 64, 32, 32, 32, num_classes]
elif number_hidden_layers == 5:
	number_neurons = [num_inputs, 64, 32, 32, 32, 32, num_classes]

training_specifics = [config.optimizer, config.epochs, config.batchsz, config.learning_rate, 0.1, 0.1, 0.1, "cross-entropy", input_data] #optimizer,epochs,batchsz,eta, gamma, beta1 &@, loss, data

#Instantitation(Creation of Neural Network)
nn = myDLkit2.feed_fwd_nn(number_neurons, activation, training_specifics, number_hidden_layers, initialization)

#wandb.watch(nn)
#print("initial Biases==>", nn.Biases)
#print("initial Weights==>", nn.Weights)
#Trainings
nn.train()
ground_truth = nn.given_label
predictions = nn.pred_label

wandb.log({"conf_mat" : wandb.plot.confusion_matrix(probs=None,y_true=ground_truth,preds=predictions,class_names=class_names)})


#print("Trained Biases==>",nn.Biases) 	
#print("Trained Weights==>", nn.Weights)
#print(nn.accuracy)