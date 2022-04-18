import numpy as np
from matplotlib import pyplot as plt


train = np.loadtxt("data/MNIST_train_images.csv", delimiter=',')
train=np.asarray(train)
train_labels = np.loadtxt("data/MNIST_train_labels.csv", delimiter=',')
train_labels=np.asarray(train_labels)
#plot 1 example from the train set 
plt.imshow(np.asarray(np.reshape(train[6, :], (28, 28))),cmap='gray', vmin=0, vmax=255)
plt.title('1 train image')
plt.savefig('1 train image.png')
