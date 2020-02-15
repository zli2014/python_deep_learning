# -*- coding: utf-8 -*-
# @File  : functions.py.py
# @Author: lizhen
# @Date  : 2020/1/27
# @Desc  : 激励函数

import numpy as np

def identity_function(x):
    '''
    恒等函数
    :param x: ndarray， 神经元的输入
    :return: 输入数据，不作任何处理
    '''
    return x

def step_function(x):
    '''
    阶段函数：大于0的部分设置为1，小于0的部分为0
    :param x:ndarray，神经元的输入
    :return:
    '''
    return np.array(x > 0, dtype=np.int)

def sigmoid(x):
    '''
    sigmoid函数
    :param x: 输入数据
    :return:
    '''
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    '''
    sigmoid的梯度
    :param x:
    :return:
    '''
    return (1.0 - sigmoid(x)) * sigmoid(x)


def relu(x):
    '''
    relu 函数
    :param x:
    :return:
    '''
    return np.maximum(0, x)

def relu_grad(x):
    '''
    relu函数的grad
    :param x:
    :return:
    '''
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def softmax(x):
    '''
    此处的softmax 有batch 版本和非batch 版本
    如果x 的维度大于2 则使用batch 版本，否则使用非batch版本
    :param x:
    :return:
    '''
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))

def sum_squared_error(y, t):
    '''

    :param y: 神经元的输出
    :param t: 标签
    :return:
    '''
    return 0.5 * np.sum((y - t) ** 2)

def mean_squared_error(y,t):
    '''
    损失函数
    :param y:
    :param t:
    :return:
    '''
    return 0.5*np.sum((y-t)**2)
def softmax_loss(X, t):
    '''
    损失函数
    :param X:
    :param t:
    :return:
    '''
    y = softmax(X)
    return cross_entropy_error(y, t)

def cross_entropy_error(y, t):
    '''
    交叉熵的损失函数
    :param y:
    :param t:
    :return:
    '''
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size



























