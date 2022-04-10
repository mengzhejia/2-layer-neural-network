from train import NeuralNetwork, getData
import pickle

x_train, t_train, x_val, t_val, x_test, t_test = getData()

results = {}
best_val = -1
best_net = None

# learning_rates = [0.01, 0.1, 0.2]
# regularization_strengths = [0.001, 0.01, 0.1]
# hidden_nodes = [25, 50, 100]
# 0.2 0.01 最好
learning_rates = [0.01, 0.1, 0.2, 0.3]
regularization_strengths = [0.001, 0.01, 0.1]
hidden_nodes = [50, 100, 128]

for lr in learning_rates:
    for reg in regularization_strengths:
        for num in hidden_nodes:
            print("本轮训练使用的超参数为：学习率 " + str(lr) + " 正则化强度 " + str(reg) + " 隐藏层节点数 " + str(num))
            net = NeuralNetwork(input_size=784, hidden_size=num, output_size=10, activation='relu')
            # 进行训练
            stats = net.train(x_train, t_train, x_val, t_val, hidden_num=num,
                          learning_rate=lr, learning_rate_decay=0.9999, batch_size=100, iters_num=10000, reg=reg, verbose=True)
            val_acc = net.accuracy(x_val, t_val)
            if val_acc > best_val:
                best_val = val_acc
                best_net = net
            results[(lr, reg, num)] = val_acc

# 打印结果
for lr, reg, num in sorted(results):
    val_acc = results[(lr, reg, num)]
    print('lr %e reg %e num %e val accuracy: %f' % (
        lr, reg, num, val_acc))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

with open('save/bestmodel', 'wb') as f:
    pickle.dump(best_net, f)

test_acc = (best_net.accuracy(x_test, t_test))
print ('Test accuracy: ', test_acc)