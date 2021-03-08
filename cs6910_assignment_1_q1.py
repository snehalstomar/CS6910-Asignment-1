'''
 Python Program for CS6910-Assignment-1-Q-1
 EE20S006-Snehal Singh Tomar
 EE20D006-Ashish Kumar
'''
#importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset
from keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

#displaying 1 image from each category
classes = {0 : 'T-shirt/top', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress', 4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag', 9 : 'Ankle Boot'}
counter = 0
for i in range(y_train.shape[0]):
  if y_train[i] in classes and counter < 10:
    ax = plt.subplot(4, 4, y_train[i]+1)
    img = x_train[i]
    ax.set_title(classes[y_train[i]])
    ax.axis('off')
    plt.tight_layout()
    plt.imshow(img, cmap='gray')
    del(classes[y_train[i]])
    counter += 1