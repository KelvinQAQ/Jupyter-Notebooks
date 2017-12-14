#! python3

from PIL import Image
from filter import filter
import matplotlib.pyplot as plt
import numpy as np


img = Image.open('/home/kelvin/Documents/Jupyter-Notebooks/Image Processing/images/moon1.png')        # 读取图像
print(img.mode, img.size)                   # 打印图像的颜色模式和尺寸信息
img_data = np.array(img)                    # 将图像像素数据单独提取出来

plt.tight_layout()
plt.subplots_adjust(left=0.08, right=1, top=0.7, bottom=0.35, wspace=0.2)

plt.subplot(131)
plt.imshow(img_data, cmap='gray')           # 绘制原始图像
plt.title('origin')

template = [-1, -1, -1, -1, 8, -1, -1, -1, -1]
img_out = filter(template, img_data)
plt.subplot(132)
plt.imshow(img_out, cmap='gray')            # 绘制滤波后的图像
plt.title('filted')

plt.subplot(133)
plt.imshow(img_out + img_data, cmap='gray') # 绘制锐化后的图像
plt.title('sharpened')

# plt.savefig('output.png')
