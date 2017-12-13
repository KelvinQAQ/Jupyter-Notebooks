#! python3

import numpy as np
# import matplotlib.pyplot as plt

def filter(t_array, img, t_size=3):
    # print(img.shape)
    w, h = img.shape
    temp = np.array(t_array, dtype='i').reshape(t_size, t_size)
    out = np.zeros((w, h), dtype='i')
    for i in range(h):
        for j in range(w):
            try:
                out[i, j] = np.sum(temp * img[i-t_size//2:i+t_size//2+1, j-t_size//2:j+t_size//2+1])
            except:
                pass
    return out

# if __name__ == '__main__':
#     img = np.random.randint(256, size=(8, 8))
#     # print(img)
#     plt.subplot(121)
#     plt.imshow(img, cmap='gray')
#     out = filter([0, -1, 0, -1, 4, -1, 0, -1, 0], img)
#     plt.subplot(122)
#     plt.imshow(out, cmap='gray')

#     plt.savefig('output.png')
#     # print(out)