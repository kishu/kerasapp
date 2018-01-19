"""
estate
보스턴 집값 예측

1. 회귀 ANN 구현
2. 학습과 평가용 데이터 불러오기
3. 회귀 ANN 학습 및 성능 평가
4. 회귀 ANN 학습 결과 분석
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import matplotlib.pyplot as plt

from keras import layers, models
from keras import datasets
from sklearn import preprocessing

class ANN(models.Model):
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')

        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = output(h)

        super().__init__(x, y)
        self.compile(loss='mse', optimizer='sgd')

def Data_func():
    (X_train, y_train), (X_test, y_test) = \
        datasets.boston_housing.load_data()
    # Transforms features by scaling each feature to a given range.
    #  feature_range : tuple (min, max), default=(0, 1)
    #  Desired range of transformed data.
    scaler = preprocessing.MinMaxScaler()
    X_train = scaler.fit_transform(X_train) #(404, 13)
    X_test = scaler.transform(X_test)

    return (X_train, y_train), (X_test, y_test)

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)
    plt.show()

def main():
    Nin = 13
    Nh = 5
    Nout = 1

    model = ANN(Nin, Nh, Nout)
    (X_train, y_train), (X_test, y_test) = Data_func()

    print('\nFit')
    history = model.fit(X_train, y_train, epochs=100, batch_size=100, validation_split=0.2)

    print('\nEvaluate')
    performance_test = model.evaluate(X_test, y_test, batch_size=100)
    
    print('\nTest Loss -> {:.2f}'.format(performance_test))

    #plot_loss(history)

if __name__ == '__main__':
    main()