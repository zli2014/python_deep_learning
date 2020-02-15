# -*- coding: utf-8 -*-
# @File  : day5.py
# @Author: lizhen
# @Date  : 2020/1/28
# @Desc  :
import numpy as np
from collections import OrderedDict
from src.test.Perceptron import Relu,numerical_gradient
from src.common.layers import Affine,SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self,
                 input_size, hidden_size, ouput_size,
                 weight_init_std=0.01):
        # init weights
        self.params={}
        self.params['W1'] = weight_init_std * np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size,ouput_size)
        self.params['b2'] = np.zeros(ouput_size)

        # gen layers
        self.layers = OrderedDict()
        self.layers['Affine1']= Affine(self.params['W1'], self.params['b1'])
        self.layers['relu'] = Relu()
        self.layers['Affine2']=Affine(self.params['W2'],self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self,x):
        '''
        正向传播
        执行每一层的forward
        :param x:
        :return:
        '''
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    def loss(self,x,t):
        '''
        loss 只有在反向传播的时候使用，因此单独写出来
        :param x:
        :param t:
        :return:
        '''
        y = self.predict(x)
        return self.lastLayer.forward(y,t)

    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 :
            t = np.argmax(t,axis=1)
        accuracy= np.sum(y ==t )/float(x.shape[0])
        return  accuracy

    def numerical_gradient(self, x,t):
        '''
        计算梯度
        :param x:
        :param t:
        :return:
        '''
        loss_W = lambda W:self.loss(x,t)
        grads={}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])#

