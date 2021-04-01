# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from src.datasets.mnist import load_mnist
from skimage import io
from src.nets.simple_convnet import SimpleConvNet

def showFeatureMap(input_mnist, network, keys='W1'):
    """

    """
    print("");
"""
SimpleConvNet:
layer1: Conv1 (W1,b1)
layer2: Relu1()
layer3: Pool1()
layer4: Affine1(W2,b2)
layer5: Relu2()
layer6: Affine(W3,b3)
loss:   SoftmaxWithLoss()
output: [10]
"""
def feature_map_show(feature_map, nx=7, margin=3, scale=10):

    C, H, W = feature_map.shape
    ny = int(np.ceil(nx))

    fig = plt.figure()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(C):
        ax = fig.add_subplot(ny, nx, i+1, xticks=[], yticks=[])
        ax.imshow(feature_map[i], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.show()

# create and load weights
network = SimpleConvNet()
network.load_params("params.pkl")
# load data
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
x = x_test[0].reshape(1,1,28,28)
# result=network.predict(x)
# y = np.argmax(result)
# print(y)
# shou input

### show featurn map
conv1=network.getLayers()['Conv1']
feature_maps = conv1.forward(x)
print("shape of feature_map :",str(feature_maps.shape)) # (1, 30, 24, 24)
print("shape of feature_maps :",feature_maps.shape)
feature_map = feature_maps.reshape(30,24,24)
print("shape of feature_map :",feature_map.shape)
feature_map_show(feature_map)
