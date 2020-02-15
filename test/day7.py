# -*- coding: utf-8 -*-
# @File  : day7.py
# @Author: lizhen
# @Date  : 2020/2/4
# @Desc  :

class BaseLayer:
    '''
    所有层的基类
    '''
    def forward(self,x,y):
        raise NotImplementedError
    def backward(self,dout):
        raise NotImplementedError
    def toString(self):
        raise NotImplementedError

class MulLayer(BaseLayer):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = x*y

        return out

    def backward(self,dout):
        '''
        反馈方面是反转x,y
        :param dout:
        :return:
        '''
        dx = dout * self.y
        dy = dout * self.x
        return  dx,dy

    def toString(self):
        print("name: Multi")
        print("x.shape %s"%str(self.x.shape))
        print("y.shape %s"%str(self.y.shape))


class AddLayer(BaseLayer):
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self,x,y):
        self.x = x
        self.y = y
        out = self.x+self.y
        return  out
    def backward(self,dout):
        dx = dout*1
        dy = dout*1
        return dx,dy
    def toString(self):
        print("name: Add")
        print("x.shape %s"%str(self.x.shape))
        print("y.shape %s"%str(self.y.shape))




