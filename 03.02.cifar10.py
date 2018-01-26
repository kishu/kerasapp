'''
컬러 이미지 분류

https://www.cs.toronto.edu/~kriz/cifar.html
총 6만장 (5만장 학습, 1만장 평가)
32(Width) x 32(Height) x 3(RGB CHANNELS)
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt

from keras import layers, models, datasets
from keras.utils import np_utils

class DNN(models.Sequential):
    def __init__(self, Nin, Nh_l, Pd_l, Nout):
        super().__init__()
        self.add(layers.Dense(Nh_l[0], activation='relu', input_shape=(Nin,), name='Hidden-1'))
        self.add(layers.Dropout(Pd_l[0]))
        self.add(layers.Dense(Nh_l[1], activation='relu', name='Hidden-2'))
        self.add(layers.Dropout(Pd_l[1]))
        # self.add(layers.Dense(Nh_l[2], activation='relu', name='Hidden-3'))
        # self.add(layers.Dropout(Pd_l[2]))
        # self.add(layers.Dense(Nh_l[3], activation='relu', name='Hidden-4'))
        # self.add(layers.Dropout(Pd_l[3]))
        # self.add(layers.Dense(Nh_l[4], activation='relu', name='Hidden-5'))
        # self.add(layers.Dropout(Pd_l[4]))
        # self.add(layers.Dense(Nh_l[5], activation='relu', name='Hidden-6'))
        # self.add(layers.Dropout(Pd_l[5]))
        # self.add(layers.Dense(Nh_l[6], activation='relu', name='Hidden-7'))
        # self.add(layers.Dropout(Pd_l[6]))
        self.add(layers.Dense(Nout, activation='softmax'))
        self.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    L, W, H, C = X_train.shape
    X_train = X_train.reshape(-1, W * H * C)
    X_test = X_test.reshape(-1, W * H * C)

    X_train = X_train / 255
    X_test = X_test / 255

    return (X_train, y_train), (X_test, y_test)

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.show()

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def main():
    Nh_l = [100, 50]
    Pd_l = [0.05, 0.5]
    # Nh_l = [320, 320, 160, 80, 40, 20, 10]
    # pd_l = [0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.001]
    Nout = 10

    (X_train, y_train), (X_test, y_test) = Data_func()
    
    model = DNN(X_train.shape[1], Nh_l, Pd_l, Nout)
    history = model.fit(X_train, y_train, epochs=30, batch_size=100, validation_split=0.2)
    performace_test = model.evaluate(X_test, y_test, batch_size=100)

    print('Test Loss and Accuracy ->', performace_test)

    plot_acc(history)
    plot_loss(history)
    plt.show()

if __name__ == '__main__':
    main()