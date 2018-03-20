import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data



if __name__ == '__main__':
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.ones([784, 10]))
    b = tf.Variable(tf.ones([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.02).minimize(cross_entropy)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    epoch = 100

    for i in range(epoch):
        batch_xs, batch_ys = mnist.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        if i%10 == 9:
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(sess.run(accuracy, feed_dict={x: mnist.test_img_data, y_: mnist.test_label_data}))
    sess.close()