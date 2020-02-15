# -*- coding: utf-8 -*-
# @File  : test_im2col.py
# @Author: lizhen
# @Date  : 2020/2/14
# @Desc  : 测试im2col
import numpy as np

from src.common.util import im2col,col2im
from src.common.layers import Convolution,Pooling


if __name__ == '__main__':
    raw_data = [3,	0,	4,	2,
                6,	5,	4,	3,
                3,	0,	2,	3,
                1,	0,	3,	1,

                1,	2,	0,	1,
                3,	0,	2,	4,
                1,	0,	3,	2,
                4,	3,	0,	1,

                4,	2,	0,	1,
                1,	2,	0,	4,
                3,	0,	4,	2,
                6,	2,	4,	5
    ]

    raw_filter=[
        1,    1,    1,    1,    1,    1,
        1,    1,    1,    1,    1,    1,
        2,    2,    2,    2,    2,   2,
        2,    2,    2,    2,    2,   2,

    ]



    input_data = np.array(raw_data)
    filter_data = np.array(raw_filter)

    x = input_data.reshape(1,3,4,4)# NCHW
    W = filter_data.reshape(2,3,2,2) # NHWC
    b = np.zeros(2)
    # b = b.reshape((2,1))
    # col1 = im2col(input_data=x,filter_h=2,filter_w=2,stride=1,pad=0)#input_data, filter_h, filter_w, stride=1, pad=0
    # print(col1)

    # print("input_data.shape=%s"%str(input_data.shape))
    # print("W.shape=%s"%str(W.shape))
    # print("b.shape=%s"%str(b.shape))
    # conv = Convolution(W,b) # def __init__(self, W, b, stride=1, pad=0)
    # out = conv.forward(x)
    # print("bout.shape=%s"%str(out.shape))
    # print(out)

    print("===================")
    pool=Pooling( pool_h=2, pool_w=2, stride=2, pad=0)
    out = pool.forward(x)
    print(out.shape)
    print(out)




