# ref. from:Tensorflow Day3 : 熟悉 MNIST 手寫數字辨識資料集: https://ithelp.ithome.com.tw/articles/10186473

import numpy as np
import matplotlib.pylab as plt
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 來看看 mnist 的型態
print (mnist.train.num_examples)
print (mnist.validation.num_examples)
print (mnist.test.num_examples)

train_img = mnist.train.images
train_label = mnist.train.labels
test_img = mnist.test.images
test_label = mnist.test.labels
print
print(" train_img 的 type : %s" % (type(train_img)))
print(" train_img 的 dimension : %s" % (train_img.shape,))
print(" train_label 的 type : %s" % (type(train_label)))
print(" train_label 的 dimension : %s" % (train_label.shape,))
print(" test_img 的 type : %s" % (type(test_img)))
print(" test_img 的 dimension : %s" % (test_img.shape,))
print(" test_label 的 type : %s" % (type(test_label)))
print(" test_label 的 dimension : %s" % (test_label.shape,))

nsample = 1
randidx = np.random.randint(train_img.shape[0], size=nsample)

for i in [0, 1, 2]:
    curr_img   = np.reshape(train_img[i, :], (28, 28)) # 28 by 28 matrix 
    curr_label = np.argmax(train_label[i, :] ) # Label
    plt.matshow(curr_img, cmap=plt.get_cmap('gray'))
    plt.title("" + str(i + 1) + "th Training Data " + "Label is " + str(curr_label))
	
 
plt.show()