import numpy as np
import myDLkit

from tensorflow.keras.datasets import fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

n_inputs = 784
n_hidden_layers = 5
n_outputs = 10
eta = 10 ** np.exp(-4)
errors = []


def img_normalized_feature_extractor(img):
	img_arr = np.zeros([1, img.shape[0]*img.shape[1]])
	for i in range(img.shape[0]):
		for j in range(img.shape[1]):
			img_arr[0, (img.shape[1]*i)+j] = img[i, j] / 255 #normalizing wrt intensities
	return img_arr

nn = myDLkit.feed_fwd_nn(n_hidden_layers, n_inputs, n_outputs)
#x_train.shape[0]/1000
for i in range(3000):
	print("iteration:", i)
	img1 = x_train[i]
	label = y_train[i]
	img_one_d = img_normalized_feature_extractor(img1)
	nn.backProp(img_one_d, label)
	errors.append(nn.error)

	for i in nn.grad_wrt_b:
		nn.layers[i].b = nn.layers[i].b - (eta*nn.grad_wrt_b[i].T)

	for i in nn.grad_wrt_W:
		nn.layers[i].W = nn.layers[i].W - (eta*nn.grad_wrt_W[i])

print(errors)