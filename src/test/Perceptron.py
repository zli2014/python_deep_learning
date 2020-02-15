# -*- coding: utf-8 -*-
# @File  : Perceptron.py
# @Author: lizhen
# @Date  : 2020/2/2
# @Desc  : 第二篇
import numpy as np


def AND(x1, x2):
    '''
    AND gate
    :param x1: must be {0,1}
    :param x2: must be {0,1}
    :return:
    '''
    w1, w2, theta = 1, 1, 2
    temp = x1 * w1 + x2 * w2

    if temp > theta:
        return 1
    else:
        return 0


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([1, 1])
    b = 0
    tmp = np.sum(x * w) + b

    if tmp < 1:
        return 0
    else:
        return 1


def Not(x1):
    w1, theta = 1, 1

    tmp = w1 * x1
    if tmp >= theta:
        return 0
    else:
        return 1


def NAND(x1, x2):
    y = AND(x1, x2)
    return Not(y)


def Sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step_func(x):
    temp = x.copy()
    temp = np.where(x > 0, temp, 0)
    temp = np.where(x <= 0, temp, 1)
    return temp


def Relu(x):
    return np.maximum(0, x)


def softmax(x):
    '''
    softmax 实现
    :param x: ndarray
    :return: y, ndarray
    '''
    C = np.max(x)
    exp_a = np.exp(x - C)
    sum_exp = np.sum(exp_a)
    y = exp_a / sum_exp
    return y


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]  # batch

    delta = 1e-7
    return -np.sum(t * np.log(y + delta)) / batch_size


def numerical_gradient(f, x):
    '''
    x的梯度
    :param x: ndarray,
    :param f:
    :return: x的梯度
    '''
    deta = 1e-4
    grad = np.zeros_like(x)  # 生成与x 形状相同的数组,用于汇总偏导数

    for idx in range(x.shape[0]):
        temp = x[idx]
        x[idx] = temp + deta  # f(x+deta)
        y1 = f(x)

        x[idx] = temp - deta  # x- deta
        y2 = f(x)

        grad[idx] = (y1 - y2) / (2 * deta)  # x+deta

        x[idx] = temp  # 赋予原来的值
    return grad


def function(x):
    return x[0] ** 2 + x[1] ** 2


def gradient_descent(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()
    return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)


class TwoLayerNet:
    def softmax(self, x):
        '''
        softmax 实现
        :param x: ndarray
        :return: y, ndarray
        '''
        C = np.max(x)
        exp_a = np.exp(x - C)
        sum_exp = np.sum(exp_a)
        y = exp_a / sum_exp
        return y

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_grad(self, x):
        return (1.0 - self.sigmoid(x)) * self.sigmoid(x)

    def cross_entropy_error(self, y, t):
        '''
        带mini-batch的交叉熵
        :param t:
        :return:
        '''
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        if t.size == y.size:
            t = t.argmax(axis=1)

        batch_size = y.shape[0]

        return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

    def __init__(self, input_size,
                 hidden_size,
                 output_size,
                 weight_init_std=0.1):
        # 初始化权重
        self.param = {}
        self.param['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.param['b1'] = np.zeros(hidden_size)
        self.param['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.param['b2'] = np.zeros(output_size)

    def predict(self, x):
        '''
        推理
        :param x:
        :return:
        '''
        W1, W2 = self.param['W1'], self.param['W2']
        b1, b2 = self.param['b1'], self.param['b2']

        out1 = np.dot(x, W1) + b1
        fx = self.sigmoid(out1)
        out2 = np.dot(fx, W2) + b2

        y = self.softmax(out2)

        return y

    def loss(self, x, t):
        '''
        计算损失值
        :param x: 输入
        :param t: 预测
        :return:
        '''
        y = self.predict(x)

        return self.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda W: self.loss(x, t)  # 计算梯度

        grads = {}
        grads['W1'] = numerical_gradient(loss_w, self.param['W1'])
        grads['b1'] = numerical_gradient(loss_w, self.param['b1'])
        grads['W2'] = numerical_gradient(loss_w, self.param['W2'])
        grads['b2'] = numerical_gradient(loss_w, self.param['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.param['W1'], self.param['W2']
        b1, b2 = self.param['b1'], self.param['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        dz1 = np.dot(dy, W2.T)
        da1 = sigmoid_grad(a1) * dz1
        grads['W1'] = np.dot(x.T, da1)
        grads['b1'] = np.sum(da1, axis=0)

        return grads


from src.datasets.mnist import *

if __name__ == '__main__':
    print(numerical_gradient(function, np.array([3.0, 4.0])))
    # min_value = gradient_descent(function, init_x=np.array([3.0,4.0]))
    # print(min_value)
