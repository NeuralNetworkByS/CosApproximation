import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#x_data = np.linspace(-1, 1, 10)

def add_layer(inputs, in_size, out_size, activation_function = None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size], seed=2000))
    biases = tf.Variable(tf.Variable(tf.zeros([1, out_size])))
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        output = Wx_plus_b
    else:
        output = activation_function(Wx_plus_b)
    return output


x_data = []
x_data_test = []
n = 30
x = 0.0
number_nuerons = 20


for i in range(n):
    x_data.append([x])
    x += 0.2
    x = float(("%0.2f" % x))

x = 0.1

for i in range(n):
    x_data_test.append([x])
    x += 0.2
    x = float(("%0.2f" % x))

y_data = np.cos(x_data)

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])
layer1 = add_layer(xs, 1, number_nuerons, activation_function = tf.nn.sigmoid) 
layer2 = add_layer(layer1, number_nuerons, number_nuerons, activation_function = tf.nn.sigmoid)
prediction = add_layer(layer2, number_nuerons, 1, activation_function = None)

loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

sess = tf.Session()

init = tf.global_variables_initializer()
sess.run(init)

err = 1
err_prev = 0
epochs = 0
while err > 0.0005:
    #err_prev = err
    _ , err, prod = sess.run([train_step, loss, prediction], feed_dict={xs: x_data, ys: y_data})
    epochs += 1
    if (epochs % 1000 == 0):
        print(epochs)
        print("error", err)


prod_test = sess.run(prediction, feed_dict={xs: x_data_test})

#print(y_data)
#print('-'*80)
#print(prod)
print('error: ', err, )
print('epochs: ', epochs)

plt.plot(x_data, y_data, label="real data")
#plt.plot(x_data, prod)
plt.plot(x_data_test, prod_test, label="aproximation")
plt.legend()
plt.show()