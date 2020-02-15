# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from src.datasets.mnist import load_mnist
from src.nets.multi_layer_net_extend import MultiLayerNetExtend
from src.common.optimizer import SGD

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)

# 获取前1000个样本作为训练数据
x_train = x_train[:1000]
t_train = t_train[:1000]

max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01


def __train(weight_init_std):
    '''
    训练
    :param weight_init_std: 初始化权重
    :return:
    '''
    # 带有bn层
    bn_network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    # 没有bn层
    network = MultiLayerNetExtend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    # 优化器
    optimizer = SGD(lr=learning_rate)
    
    train_acc_list = []
    bn_train_acc_list = []
    
    iter_per_epoch = max(train_size / batch_size, 1)
    epoch_cnt = 0
    
    for i in range(1000000000):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]
    
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
    
        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
                
    return train_acc_list, bn_train_acc_list


# 3.绘图
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(max_epochs)

for i, w in enumerate(weight_scale_list):#
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w) # 训练网络
    
    # plt.subplot(4,4,i+1)
    plt.title("W:%3f"%(w))
    # if i == 15:
    #     # 绘制给各自的子图
    #     plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
    #     plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    # else:
    #     plt.plot(x, bn_train_acc_list, markevery=2)
    #     plt.plot(x, train_acc_list, linestyle="--", markevery=2)


    # if i % 4:
    #     plt.yticks([])
    #
    # else:
    #     plt.ylabel("accuracy")
    # if i < 12:
    #     plt.xticks([])
    #
    # else:
    #     plt.xlabel("epochs")
    plt.plot(x, bn_train_acc_list, label=str(i)+'epoch Batch Normalization', markevery=2)
    plt.plot(x, train_acc_list, linestyle="--", label=str(i)+'epoch Normal(without BatchNorm)', markevery=2)

    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(loc='lower right')
plt.savefig("test"+str(i)+".jpg")
