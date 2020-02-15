# -*- coding: utf-8 -*-
# @File  : layers.py
# @Author: lizhen
# @Date  : 2020/1/29
# @Desc  : 神经网络相关的层

import numpy as np
from test import functions


class Relu:
    def __init__(self):
        '''
        Relu class: mask是有0/1 构成的numpy.ndarray:
        它会把正向传播的输入x中>=0的元素保存为0，其他地方保持不变

        反向传播的时候会使用此次的mask,将dout中对应的true的位置设置为0
        '''
        self.mask = None
    def forward(self, x):
        '''
        将x中小于0的数值设置为0， 大于0的部分不变
        :param x:  ndarry
        :return:
        '''
        self.mask = (x<=0) # 记录小于0的index
        out = x.copy() # 值copy
        out[self.mask] = 0
        return out
    def backward(self,dout):
        '''
        对 dout 做微分
        :param dout:
        :return:
        '''
        dout[self.mask] = 0
        dx = dout
        return  dx


class Sigmoid:
    def __init__(self):
        '''
        根据推到的公式，发现dy会在反向传播的时候用到，
        因此，将y 设置为 成员变量
        '''
        self.out = None
    def forward(self,x):
        out = 1/(1+ np.exp(-x))
        self.out = out

    def backward(self,dout):
        dx = dout*(1.0 - self.out)*self.out
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

        self.dW = None
        self.db = None
    def forward(self,x):
        self.x = x
        out = np.dot(x,self.W) + self.b
        return out

    def backward(self,dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        return  dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.x = None
    def forward(self, x, t):
        self.t = t
        self.y = functions.softmax(x)
        self.loss = functions.cross_entropy_error(self.y, self.t)

        return  self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.x - self.t ) / batch_size

        return dx
