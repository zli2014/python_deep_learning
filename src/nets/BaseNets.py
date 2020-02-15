# -*- coding: utf-8 -*-
# @File  : Nets.py
# @Author: lizhen
# @Date  : 2020/2/15
# @Desc  : 网络层的基类
class Net:
    def loss(self, x, t):
        '''
        调用优化器opt, 计算 x 与t 之间的差距
        :param x:
        :param t:
        :return:
        '''
        pass
    def gradient(self, x, t):
        pass;
    def numerical_gradient(self, x, t):
        """
        调用loss(),获取loss value
        根据loss值，计算数值微分，
        :param x:
        :param t:
        :return:
        """
        pass

