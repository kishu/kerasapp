"""
CNN for cifar
32(Width) x 32(Height) x 3(RGB CHANNELS)
"""

import os
import datetime
import numpy as np
import matplotlib.pyplot as plt

from keras import backend, datasets
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.utils import np_utils

from sklearn import model_selection, metrics
from sklearn.preprocessing import MinMaxScaler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CNN(Model):
    def __init__(self, nb_classes, in_shape=None):
        self.nb_classes = nb_classes
        self.in_shape = in_shape
        self.build_model()
        super().__init__(self.x, self.y)
        self.compile()

    def build_model(self):
        nb_classes = self.nb_classes
        in_shape = self.in_shape

        x = Input(in_shape)
        h = Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=in_shape)(x) # 29 x 29 x 32
        h = Conv2D(64, kernel_size=(3, 3), activation='relu')(h) # 27 x 27 x 64
        h = MaxPooling2D(pool_size=(2, 2))(h)
        h = Dropout(0.25)(h)
        h = Flatten()(h)
        z_cl = h

        h = Dense(128, activation='relu')(h)
        h = Dropout(0.5)(h)
        z_fl = h

        y = Dense(nb_classes, activation='softmax', name='preds')(h)

        self.cl_part = Model(x, z_cl)
        self.fl_part = Model(x, z_fl)

        self.x, self.y = x, y

    def compile(self):
        Model.compile(self, loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

class DataSet():
    def __init__(self, X, y, nb_classes, scaling=True, test_size=0.2, random_state=0):
        self.X = X
        self.add_channels()
        X = self.X

        # 80% 학습데이터, 20% 테스트 데이터
        # random_state is the seed used by the random number generator
        X_train, X_test, y_train, y_test = \
            model_selection.train_test_split(X, y, test_size=test_size, random_state=random_state)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        if scaling:
            scaler = MinMaxScaler()

            n = X_train.shape[0] # 이미지 갯수
            X_train_reshape = X_train.reshape(n, -1) # (n, 32, 32, 3) => (n, 3072)
            X_train_transform = scaler.fit_transform(X_train_reshape) # [59. 62. 63. ..., => [0.23137257 0.24313727 0.24705884 ...,
            X_train = X_train_transform.reshape(X_train.shape) # (n, 3072) => (n, 32, 32, 3)

            n = X_test.shape[0]
            X_test = scaler.transform(X_test.reshape(n, -1)).reshape(X_test.shape)

            self.scaler = scaler

            y_train = np_utils.to_categorical(y_train, nb_classes)
            y_test = np_utils.to_categorical(y_test, nb_classes)

            self.X_train, self.X_test = X_train, X_test
            self.y_train, self.y_test = y_train, y_test

    def add_channels(self):
        X = self.X

        if len(X.shape) == 3: # if image is gray
            N, img_rows, img_cols = X.input_shape
            # It specifies which dimension ordering convention Keras will follow.
            # tf - "tensorflow" or th "theano".
            if(backend.image_dim_ordering() == "th"):
                X = X.reshape(X.shape[0], 1, img_rows, img_cols)
                input_shape = (1, img_rows, img_cols)
            else:
                X = X.reshape(X.shape[0], img_rows, img_cols, 1)
                input_shape = (img_rows, img_cols, 1)
        else:
            input_shape = X.shape[1:] # (32, 32, 3)

        self.X = X
        self.input_shape = input_shape

class Machine():
    """
    분류 CNN의 학습 및 성능 평가를 위한 머신 클래스
    """
    def __init__(self, X, y, nb_classes=2, fig=True):
        self.nb_classes = nb_classes
        self.set_data(X, y)
        self.set_model()
        self.fig = fig

    def set_data(self, X, y):
        nb_classes = self.nb_classes
        self.data = DataSet(X, y, nb_classes)

    def set_model(self):
        nb_classes = self.nb_classes
        data = self.data
        self.model = CNN(nb_classes=nb_classes, in_shape=data.input_shape)

    def fit(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model

        history = model.fit(data.X_train, data.y_train,
            batch_size=batch_size, epochs=nb_epoch, verbose=verbose,
            validation_data=(data.X_test, data.y_test))

        return history

    def run(self, nb_epoch=10, batch_size=128, verbose=1):
        data = self.data
        model = self.model
        fig = self.fig

        history = self.fit(nb_epoch=nb_epoch, batch_size=batch_size, verbose=verbose)
        score = model.evaluate(data.X_test, data.y_test, verbose=0)

        print('Coufusion matrix')

        Y_test_pred = model.predict(data.X_test, verbose=0)
        y_test_pred = np.argmax(Y_test_pred, axis=1)
        y_test = np.argmax(data.y_test, axis=1)

        print(metrics.confusion_matrix(y_test, y_test_pred))

        print('Test score:', score[0])
        print('Test accuracy:', score[1])

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        foldname = 'output_' + suffix
        os.makedirs(foldname)
        np.save(os.path.join(foldname, 'history_history.npy'), history.history)
        model.save_weights(os.path.join(foldname, 'dl_model.h5'))
        print('Output results are saved in ', foldname)

        if fig:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 2, 1)
            plt.title('Model Accuracy')
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend(['Train', 'Test'], loc=0)

            plt.subplot(1, 2, 2)
            plt.title('Model Loss')
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend(['Train', 'Test'], loc=0)

            plt.show()

        self.history = history

def main():
    (X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
    m = Machine(X_train, y_train, nb_classes=10)
    m.run()

if __name__ =='__main__':
    main()
