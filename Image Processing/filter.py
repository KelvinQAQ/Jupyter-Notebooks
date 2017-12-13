#! python3

import numpy as np

def filter(t_array, img, t_size=3):
    print(img.shape)
    # w, h = img.shape
    # temp = np.array(t_array, dtype='i').reshape(t_size)
    # out = np.zeros((h, w), dtype='i')
    # for i in range(h):
    #     for j in range(w):
    #         out[i, j] = np.sum(temp * img[i-t_size//2:i+t_size//2+1, j-t_size//2:j+t_size//2+1])
    # return out

if __name__ == '__main__':
    filter([0, -1, 0, -1, 4, -1, 0, -1, 0], np.random.randint(256, size=(8, 8))