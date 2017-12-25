import numpy as np

img_data = open('t10k-images.idx3-ubyte')
img_label = open('t10k-labels.idx1-ubyte')

num_test = 10000
num_train = 60000
length = 28 * 28

array_test_data = np.fromfile(img_data, dtype='b')
array_test_label = np.fromfile(img_label, dtype='b')

print(array_test_data[:500])
# print(array_test_data.size)

imgs = []

for i in range(num_test):
    imgs.append(array_test_data[16 + i * length : 16 + (i + 1) * length].reshape(28, 28))
print(imgs[0])