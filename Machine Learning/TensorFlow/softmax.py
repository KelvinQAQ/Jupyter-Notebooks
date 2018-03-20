import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

class mnist(object):

    @staticmethod
    def next_batch(tp='tr', length=100):
        if tp == 'tr':
            index = np.random.random_integers(0, mnist.train_img_data.shape[0], length)
            img_batch = mnist.train_img_data[index, :]
            label_batch = mnist.train_label_data[index, :]
        elif tp == 't':
            index = np.random.random_integers(0, mnist.test_img_data.shape[0], length)
            img_batch = mnist.test_img_data[index, :]
            label_batch = mnist.test_label_data[index, :]

        return img_batch, label_batch

    @staticmethod
    def load_dataset(one_hot=True):
        all_path = ['MNIST/t10k-images.idx3-ubyte',
                    'MNIST/t10k-labels.idx1-ubyte', 
                    'MNIST/train-images.idx3-ubyte', 
                    'MNIST/train-labels.idx1-ubyte']
        mnist.train_img_data = mnist.load_data(all_path[2], 'im', one_hot=one_hot)
        mnist.train_label_data = mnist.load_data(all_path[3], 'lb', one_hot=one_hot)
        mnist.test_img_data = mnist.load_data(all_path[0], 'im', one_hot=one_hot)
        mnist.test_label_data = mnist.load_data(all_path[1], 'lb', one_hot=one_hot)

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
            data = raw_data[16:].reshape((-1,784))
        elif tp == 'lb':  # load label data
            data = raw_data[8:].reshape((-1,1))
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

if __name__ == '__main__':
    mnist.load_dataset(one_hot=True)

    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.ones([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    y_ = tf.placeholder(tf.float32, [None, 10])
    cross_entropy = -tf.reduce_sum(y_*tf.log(y))
    train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    # square_loss = tf.reduce_sum(tf.square(y_ - y))
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(square_loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    epoch = 1000

    for i in range(epoch):
        batch_xs, batch_ys = mnist.next_batch(length=128)
        _, x_return, W_return = sess.run([train_step, x, W], feed_dict={x: batch_xs/255.0, y_: batch_ys})

        if i%10 == 9:
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print(sess.run(accuracy, feed_dict={x: mnist.test_img_data, y_: mnist.test_label_data}))
            # print(np.sum(x_return))
            # print(np.sum(W_return))
    sess.close()

# print(mnist.next_batch('tr', 10))
# print(mnist.next_batch('t', 10))
