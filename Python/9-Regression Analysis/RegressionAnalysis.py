#reference to https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/2-2-example2/

import tensorflow as tf
import numpy as np

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.5 + 0.6  #target

Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases #predict function

loss = tf.reduce_mean(tf.square(y-y_data)) 
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# init = tf.initialize_all_variables()  
init = tf.global_variables_initializer()  

sess = tf.Session()
sess.run(init)          # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))