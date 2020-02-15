# -*- coding: utf-8 -*-
# @File  : utile.py
# @Author: lizhen
# @Date  : 2020/1/27
# @Desc  : 工具
import numpy as np


def smooth_curve(x):
    """
    用于平滑损失函数图

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]


def shuffle_dataset(x, t):
    """
    对数shaffle
    :param x: 训练数据
    :param t:标签
    :return: shuffle后的x,t
    """

    permutation = np.random.permutation(x.shape[0])# 随机
    x = x[permutation,:] if x.ndim == 2 else x[permutation,:,:,:]
    t = t[permutation]

    return x, t

def conv_output_size(input_size, filter_size, stride=1, pad=0):
    """
    计算conv的输出大小
    :param input_size: 输入的大小,
    :param filter_size:filter的大小，
    :param stride:
    :param pad:
    :return:
    """
    return (input_size + 2*pad - filter_size) / stride + 1


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    将4D tensor 转换成2D 矩阵
    :param input_data: 输入数据由4维数组组成（N,C，H,W）
    :param filter_h:   filer的高
    :param filter_w:   filter的宽
    :param stride:     stride
    :param pad:        padding
    :return:           2D矩阵
    """
    # 计算输出的大小
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    # padding
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # 计算单元
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    # 重新排列
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col



def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """
    将2D矩阵转换成4D tensor
    :param col:
    :param input_shape: 输入的形状
    :param filter_h:
    :param filter_w:
    :param stride:
    :param pad:
    :return: 4D的tensor
    """

    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]