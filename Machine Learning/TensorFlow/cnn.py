import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class mnist(object):

    @staticmethod
    def next_batch(tp='tr', length=100):
        if tp == 'tr':
            index = np.random.randint(0, mnist.train_img_data.shape[0], length)
            img_batch = mnist.train_img_data[index, :]
            label_batch = mnist.train_label_data[index, :]
        elif tp == 't':
            index = np.random.randint(0, mnist.test_img_data.shape[0], length)
            img_batch = mnist.test_img_data[index, :]
            label_batch = mnist.test_label_data[index, :]

        return img_batch, label_batch

    @staticmethod
    def load_dataset(one_hot=True):
        all_path = ['MNIST/t10k-images.idx3-ubyte',
                    'MNIST/t10k-labels.idx1-ubyte',
                    'MNIST/train-images.idx3-ubyte',
                    'MNIST/train-labels.idx1-ubyte']
        mnist.train_img_data = mnist.load_data(
            all_path[2], 'im', one_hot=one_hot)
        mnist.train_label_data = mnist.load_data(
            all_path[3], 'lb', one_hot=one_hot)
        mnist.test_img_data = mnist.load_data(
            all_path[0], 'im', one_hot=one_hot)
        mnist.test_label_data = mnist.load_data(
            all_path[1], 'lb', one_hot=one_hot)

    @staticmethod
    def load_data(path, tp='im', length=None, one_hot=True):
        """
        path: file path
        tp: data type, 'im' image data, 'lb' label data
        length: length of the data to load
        """
        with open(path, 'r') as _file:
            raw_data = np.fromfile(_file, dtype=np.uint8)
        if tp == 'im':    # load image data
            data = raw_data[16:].reshape((-1, 784))
        elif tp == 'lb':  # load label data
            data = raw_data[8:].reshape((-1, 1))
            if one_hot:   # if label data is one-hot shape, convert
                data_shape = data.shape
                one_hot_data = np.zeros((data_shape[0], 10))
                for i in range(data_shape[0]):
                    one_hot_data[i, data[i, 0]] = 1
                data = one_hot_data

        if length != None:
            return data[:length, :]
        else:    # return whole dataset default
            return data


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':
    mnist.load_dataset(one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    # 第一层卷积层
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积层
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # 全连接层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 评估
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    epoch = 1000

    for i in range(epoch):
        batch_xs, batch_ys = mnist.next_batch(length=50)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 1.0})
        if i % 10 == 0:
            test_batch_xs, test_batch_ys = mnist.next_batch('t', length=100)
            acc = sess.run(accuracy, feed_dict={x: test_batch_xs, y_: test_batch_ys, keep_prob: 1.0})
            print(acc)
