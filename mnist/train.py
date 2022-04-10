import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation=None, lamb=0.01):
        self.parameters = {}
        self.parameters['w1'] = 1e-3 * np.random.rand(input_size, hidden_size)
        self.parameters['b1'] = np.zeros(hidden_size)
        self.parameters['w2'] = 1e-3 * np.random.rand(hidden_size, output_size)
        self.parameters['b2'] = np.zeros((output_size))
        self.activation = activation
        self.lamb = lamb

    def activate(self, res):
        '''激活函数'''
        if self.activation == 'relu':
            return np.maximum(0, res)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-res))
        elif self.activation == 'tanh':
            return np.tanh(res)
        elif self.activation == None:
            return res

        return res

    def activateDerivative(self, res):
        '''激活函数求导'''
        if self.activation == 'relu':
            gradient = np.array(res, copy=True)
            gradient[res > 0] = 1
            gradient[res <= 0] = 0
            return gradient
        elif self.activation == 'sigmoid':
            return res * (1 - res)
        elif self.activation == 'tanh':
            return 1 - res * res
        elif self.activation == None:
            return np.ones_like(res)

        return res

    def forward(self, X):
        w1, b1 = self.parameters['w1'], self.parameters['b1']
        w2, b2 = self.parameters['w2'], self.parameters['b2']

        z1 = np.dot(X, w1) + b1
        a1 = self.activate(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = softmax(z2)

        return a2

    def l2Regularizaion(self, x, lamb):
        return lamb * (np.sum(self.parameters['w1'] ** 2) +
                       np.sum(self.parameters['w2'] ** 2)) / x.shape[0]

    def loss(self, x, t):
        y = self.forward(x)
        loss = np.sum((y - t) ** 2) / 2 + self.l2Regularizaion(x, self.lamb)

        return loss

    def accuracy(self, x, t):
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def gradient(self, X, t):
        w1, w2 = self.parameters['w1'], self.parameters['w2']
        b1, b2 = self.parameters['b1'], self.parameters['b2']
        grads = {}

        batch_num = X.shape[0]  # 把输入的所有列一起计算，因此可以快速

        z1 = np.dot(X, w1) + b1
        a1 = self.activate(z1)
        z2 = np.dot(a1, w2) + b2
        a2 = softmax(z2)

        dz2 = (a2 - t) / batch_num
        grads['w2'] = np.dot(a1.transpose(), dz2)
        grads['b2'] = np.sum(dz2, axis=0)
        dz1 = np.dot(dz2, w2.transpose()) * self.activateDerivative(a1)
        # da1 = sigmoid_grad(a1) * dz1
        grads['w1'] = np.dot(X.transpose(), dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads

    def train(self, x_train, t_train, x_val, t_val, hidden_num=50,
              learning_rate=0.1, learning_rate_decay=0.9999, batch_size=100, iters_num=10000, reg=0.01, verbose=False):
        # start = time.perf_counter()  # 程序计时

        # (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

        train_loss_list = []
        train_acc_list = []
        val_acc_list = []

        v_w2, v_b2 = 0.0, 0.0
        v_w1, v_b1 = 0.0, 0.0

        # 超参数
        train_size = x_train.shape[0]
        # iters_num = 10000
        # batch_size = 100
        # learning_rate = 0.1
        # learning_rate_decay = 0.0001

        iter_per_epoch = max(train_size / batch_size, 1)

        network = NeuralNetwork(input_size=784, hidden_size=hidden_num, output_size=10, activation='relu', lamb=reg)

        if verbose:
            print("使用默认的超参数训练，其中学习率为0.2，正则化强度为0.01，隐藏层节点个数为50")

        for i in range(iters_num):
            # 获取mini-batch
            batch_mask = np.random.choice(train_size, batch_size)
            x_batch = x_train[batch_mask]
            t_batch = t_train[batch_mask]

            # 计算梯度
            # grad = network.numerical_gradient(x_batch, t_batch)
            grad = network.gradient(x_batch, t_batch)

            # 更新参数
            for key in ('w1', 'b1', 'w2', 'b2'):
                network.parameters[key] -= learning_rate * grad[key]

            # 记录学习过程的损失变化
            loss = network.loss(x_batch, t_batch)
            train_loss_list.append(loss)

            # SGD结合向量
            v_w2 = 0.9 * v_w2 - learning_rate * grad['w2']
            self.parameters['w2'] += v_w2
            v_b2 = 0.9 * v_b2 - learning_rate * grad['b2']
            self.parameters['b2'] += v_b2
            v_w1 = 0.9 * v_w1 - learning_rate * grad['w1']
            self.parameters['w1'] += v_w1
            v_b1 = 0.9 * v_b1 - learning_rate * grad['b1']
            self.parameters['b1'] += v_b1

            if i % iter_per_epoch == 20:
                learning_rate *= learning_rate_decay

            if verbose and i % iter_per_epoch == 0:
                train_acc = network.accuracy(x_train, t_train)
                val_acc = network.accuracy(x_val, t_val)
                train_acc_list.append(train_acc)
                val_acc_list.append(val_acc)
                print('iteration %d loss %f, train_acc: %f, val_acc: %f'
                      % (i, loss, train_acc, val_acc))
        # 画损失函数的变化
        x1 = np.arange(len(train_loss_list))
        ax1 = plt.subplot(121)
        plt.plot(x1, train_loss_list)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title('lr %f reg %f num %f' % (learning_rate, reg, hidden_num))

        # 画训练精度，测试精度随着epoch的变化
        markers = {'train': 'o', 'val': 's'}
        x2 = np.arange(len(train_acc_list))
        ax2 = plt.subplot(122)
        plt.plot(x2, train_acc_list, label='train acc')
        plt.plot(x2, val_acc_list, label='val acc', linestyle='--')
        plt.xlabel("epochs")
        plt.ylabel("accuracy")
        plt.ylim(0, 1.0)
        plt.legend(loc='lower right')
        plt.show()

        return {
            'loss_history': train_loss_list,
            'train_acc_history': train_acc_list,
            'val_acc_history': val_acc_list,
        }


def getData(num_training= 50000, num_validation=10000, num_test=10000):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    t_val = t_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    t_train = t_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    t_test = t_test[mask]

    # 预处理数据
    mean_image = np.mean(x_train, axis=0)
    x_train -= mean_image
    x_val -= mean_image
    x_test -= mean_image

    # 拉伸
    x_train = x_train.reshape(num_training, -1)
    x_val = x_val.reshape(num_validation, -1)
    x_test = x_test.reshape(num_test, -1)

    return x_train, t_train, x_val, t_val, x_test, t_test


x_train, t_train, x_val, t_val, x_test, t_test = getData()
print('Train data shape: ', x_train.shape)
print('Train labels shape: ', t_train.shape)
print('Validation data shape: ', x_val.shape)
print('Validation labels shape: ', t_val.shape)
print('Test data shape: ', x_test.shape)
print('Test labels shape: ', t_test.shape)


net = NeuralNetwork(input_size=784, hidden_size=50, output_size=10, activation='relu')

stats = net.train(x_train, t_train, x_val, t_val,
                  learning_rate=0.2, learning_rate_decay=0.9999, batch_size=100, iters_num=10000, reg=0.01, verbose=True)

val_acc = net.accuracy(x_val, t_val)
print('Validation accuracy: ', val_acc)





# 还需要做的：写实验报告，建github，模型保存好上传百度云
