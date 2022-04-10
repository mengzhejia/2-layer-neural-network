import pickle
from matplotlib import pyplot as plt
import numpy as np

with open('save/bestmodel', 'rb') as f:
    model = pickle.load(f)

print(model)



def show_net_weights(net):
    w1 = net.parameters['w1']
    # W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(w1)
    plt.gca().axis('off')
    plt.savefig('w1.png')
    plt.show()
    w2 = net.parameters['w2']
    # W1 = W1.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(w2)
    plt.gca().axis('off')
    plt.savefig('w2.png')
    plt.show()


show_net_weights(model)
