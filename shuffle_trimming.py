# -*- coding: utf-8 -*-
"""Untitled5.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FRgGBB3Zt8cLieMwSxOZaeh9rcrUMCsI
"""

import numpy as np
import random
from keras.datasets import fashion_mnist

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

hypo = 5000        # training exmaples need to be sampled
examples = X_train.shape[0]
samples = random.sample(range(0,examples),hypo)     #samples training

xs_train = X_train[samples]     #samples
ys_train = y_train[samples]



