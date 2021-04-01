# -*- coding: utf-8 -*-
# @File  : show_mnist.py
# @Author: lizhen
# @Date  : 2020/1/27
# @Desc  : 显示图片

from src.datasets.mnist import load_mnist
from skimage import io


def img_show(data):
    # pil_img = Image.fromarray(np.uint8(data))
    io.imshow(data)
    io.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)
print(img.shape)

img_show(img)