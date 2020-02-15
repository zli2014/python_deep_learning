# -*- coding: utf-8 -*-
# @File  : show_3Dplt.py
# @Author: lizhen
# @Date  : 2020/2/12
# @Desc  :
from matplotlib import pyplot as plot  # 用来绘制图形
import numpy as np  # 用来处理数据
from mpl_toolkits.mplot3d import Axes3D  # 用来给出三维坐标系。
figure = plot.figure()

def f(x, y):
    return x**2 / 20.0 + y**2

# 画出三维坐标系：
axes = Axes3D(figure)
X = np.arange(-10, 10, 0.25)
Y = np.arange(-10, 10, 0.25)
X, Y = np.meshgrid(X, Y) # 限定图形的样式是网格线的样式：
Z = f(X, Y )
axes.plot_surface(X, Y, Z, cmap='rainbow') # 绘制曲面，采用彩虹色着色：
# 图形可视化：
plot.show()


