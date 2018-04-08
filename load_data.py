from keras.datasets import cifar10
from matplotlib import pyplot
from scipy.misc import toimage

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

for i in range(0,9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(toimage(X_train[i]))

pyplot.show()
