# -*- coding: utf-8 -*-
# @File  : layers.py
# @Author: lizhen
# @Date  : 2020/1/27
# @Desc  : 实现网络层
from src.common.functions import *
from src.common.util import im2col, col2im

class BaseLayer:
    def forward(self,x):
        pass
    def backward(self, dout):
        pass


class Relu(BaseLayer):
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


class Sigmoid(BaseLayer):
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx


class Affine(BaseLayer):
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 计算微分
        self.dW = None
        self.db = None

    def forward(self, x):
        # 为了支持支持张量的计算，将x先做形状修改 相当于transOp
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)

        # out = w*x+b
        self.x = x
        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):

        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        # 将dx形状trans回去（张量支持）
        dx = dx.reshape(*self.original_x_shape)
        return dx


class SoftmaxWithLoss(BaseLayer):
    def __init__(self):
        self.loss = None
        self.y = None  # softmax的输出
        self.t = None  # 标签函数

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size:  # 处理one-hot
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size

        return dx


class Dropout(BaseLayer):
    """
    http://arxiv.org/abs/1207.0580
    """
    def __init__(self, dropout_ratio=0.5):
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        '''
        注意：如果forward的个别神经元失活，也就代表着在反馈的时候一些神经元失活，应该写在backward里面，
        但是还不知道怎么实现，也有可能是我理解的不对
        :param dout:
        :return:
        '''
        return dout * self.mask


class BatchNormalization(BaseLayer):
    """
    http://arxiv.org/abs/1502.03167
    """
    def __init__(self, gamma, beta, momentum=0.9, running_mean=None, running_var=None):
        self.gamma = gamma
        self.beta = beta
        self.momentum = momentum
        self.input_shape = None  # 转换层为4D，全连接层为2D

        # 平均值和方差
        self.running_mean = running_mean
        self.running_var = running_var

        # backward时使用的中间结果
        self.batch_size = None
        self.xc = None
        self.std = None
        self.dgamma = None
        self.dbeta = None

    def forward(self, x, train_flg=True):
        self.input_shape = x.shape
        if x.ndim != 2:
            N, C, H, W = x.shape
            x = x.reshape(N, -1)

        out = self.__forward(x, train_flg)

        return out.reshape(*self.input_shape)

    def __forward(self, x, train_flg):
        if self.running_mean is None:
            N, D = x.shape
            self.running_mean = np.zeros(D)
            self.running_var = np.zeros(D)

        if train_flg:
            mu = x.mean(axis=0)
            xc = x - mu
            var = np.mean(xc ** 2, axis=0)
            std = np.sqrt(var + 10e-7)
            xn = xc / std

            self.batch_size = x.shape[0]
            self.xc = xc
            self.xn = xn
            self.std = std
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            xc = x - self.running_mean
            xn = xc / ((np.sqrt(self.running_var + 10e-7)))

        out = self.gamma * xn + self.beta
        return out

    def backward(self, dout):
        if dout.ndim != 2:
            N, C, H, W = dout.shape
            dout = dout.reshape(N, -1)

        dx = self.__backward(dout)

        dx = dx.reshape(*self.input_shape)
        return dx

    def __backward(self, dout):
        dbeta = dout.sum(axis=0)
        dgamma = np.sum(self.xn * dout, axis=0)
        dxn = self.gamma * dout
        dxc = dxn / self.std
        dstd = -np.sum((dxn * self.xc) / (self.std * self.std), axis=0)
        dvar = 0.5 * dstd / self.std
        dxc += (2.0 / self.batch_size) * self.xc * dvar
        dmu = np.sum(dxc, axis=0)
        dx = dxc - dmu / self.batch_size

        self.dgamma = dgamma
        self.dbeta = dbeta

        return dx


class Convolution(BaseLayer):
    def __init__(self, W, b, stride=1, pad=0):
        '''
        conv的构造函数
        :param W: 2D矩阵
        :param b:
        :param stride:
        :param pad:
        '''
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间结果（backward的时候使用）
        self.x = None
        self.col = None
        self.col_W = None

        # 权重的梯度/偏置的梯度
        self.dW = None
        self.db = None

    def forward(self, x):
        '''
        使用im2col 将输入的x 转换成2D矩阵
        然后 y= w*x+b 以矩阵的形式完成
        最后返回y
        :param x: x为4D tensor, 输入数据
        :return: out=w*x+b
        '''
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2 * self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2 * self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        print("col.shape=%s"%str(col.shape))
        print("col_W.shape=%s"%str(col_W.shape))

        out = np.dot(col, col_W)
        print("out.shape=%s"%str(out.shape))
        out=out+ self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        '''
        反馈过程中也需要将2D 矩阵转换为4D tensor
        :param dout: 梯度差
        :return:
        '''
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN) # NCHW

        self.db = np.sum(dout, axis=0)# NHWC ， 求和
        self.dW = np.dot(self.col.T, dout) # 点乘w
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling(BaseLayer):
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad

        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h * self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)

        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))# 填充0
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()# 填充dout
        dmax = dmax.reshape(dout.shape + (pool_size,))

        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)

        return dx
