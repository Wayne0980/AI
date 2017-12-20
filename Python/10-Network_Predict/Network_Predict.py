import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels

def sigmoid(x):
	return 1/(1+np.exp(-x))
	
def init_work():
	network = pickle.load(open("sample_weight.pkl",'rb'))
	return network;

def predict(network,x):
	W1,W2,W3 = network['W1'],network['W2'],network['W3']
	b1,b2,b3 = network['b1'],network['b2'],network['b3']
	
	a1 = np.dot(x,W1)+b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1,W2)+b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2,W3)+b3
	y = sigmoid(a3)
	return y
	
network = init_work()

accuracy_cnt = 0
for i in range(len(test_img)):
	y = predict(network,test_img[i])
	p = np.argmax(y)
	if p == test_label[i]:
		accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt)/len(test_img)))
