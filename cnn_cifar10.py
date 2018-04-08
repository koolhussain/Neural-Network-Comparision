from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, TensorBoard
import numpy as np

epochs = 25
lr = 0.01
seed = 7
np.random.seed(seed)

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

def simple_cnn():
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(32,32,3), padding='same',
                     activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3,3), padding='same', activation='relu',
                     kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.summary()

    return model

def train(model):
    filepath="C:\\Users\\koolhussain\\Desktop\\New project\\Models\\simple_cnn\\simple_cnn-{epoch:02d}-{val_acc:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor="val_acc", verbose=0,
                                 save_best_only=True, mode="auto")
    tensorboard = TensorBoard(log_dir="./logs/simple_cnn", write_graph=True)
    sgd = SGD(lr=lr, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,
                  metrics=["accuracy"])
    model.fit(X_train, y_train, validation_data=(X_test, y_test),
              epochs=epochs, batch_size=32, callbacks=[checkpoint, tensorboard],
              verbose=1)
    score = model.evaluate(X_test, y_test, verbose=0)
    print(score)

def main():
    model = simple_cnn()
    train(model)

if __name__ == "__main__":
    main()
    
    
