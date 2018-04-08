from keras.datasets import cifar10
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import np_utils
import numpy as np

seed = 7
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()


num_pixels = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
X_train = X_train.reshape(-1, num_pixels).astype("float32")
print(X_train.shape)
X_test = X_test.reshape(-1, num_pixels).astype("float32")
print(X_test.shape)

print(X_train[0])
print(X_test[0])

X_train = X_train / 255
X_test = X_test / 255

print(X_train[0])
print(X_test[0])

print(y_train[0])
print(y_test[0])

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print(num_classes)
print(y_train[0])
print(y_test[0])

def simple_mlp():
    model = Sequential()
    model.add(Dense(num_pixels, input_dim=num_pixels,
                    kernel_initializer="normal", activation="relu"))
    model.add(Dense(num_classes, kernel_initializer="normal",
                    activation="softmax"))

    model.summary()

    return model

def train(model):
    filepath="C:\\Users\\koolhussain\\Desktop\\New project\\Models\\simple_mlp\\simple_mlp-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor="val_acc",
                                 verbose=0,
                                 save_best_only= True,
                                 mode="auto")
    tensorboard = TensorBoard(log_dir='./logs/simple_mlp',
                              write_graph=True)
    ##C:\Users\koolhussain\Desktop\New project>tensorboard --logdir=logs/simple_mlp
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=100, batch_size=200, callbacks=[checkpoint, tensorboard],
              verbose=1)
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)

def main():
    model = simple_mlp()
    train(model)

if __name__ == "__main__":
    main()
              
              
