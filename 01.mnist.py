"""
ANN
"""
import numpy as np
import matplotlib.pyplot as plt
from keras import layers, models, datasets
from keras.utils import np_utils

# 신경망 구조 정의

# 분산 방식 모델링을 포함하는 함수형 구현
# ANN 모델을 분산 방식으로 구현한다. 모델 구현은 함수형 방식을 사용한다.
def ANN_models_func(Nin, Nh, Nout):
    # 입력계층 정의
    x = layers.Input(Shape=(Nin,))

    # 은닉계층 구조와 수 정의
    # ReLU f(x) = max(x, 0)
    h = layers.Activation('relu')(layers.Dense(Nh)(x))
    
    # 출력계층 정의
    y = layers.Activation('softmax')(layers.Dense(Nout)(h))

    # 모델 생성. 입력과 출력을 지정해 모델을 생성한다.
    # 딥러팅 구조가 여러가지 딥러닝에 필요한 함수와 연계되도록 만드는 역할
    model = models.Model(x, y)

    # 컴파일. 엔진, CPU/GPU 등에 따라 다양한 초기화 및 최적화 한다.
    # metric: 성능 검증 위해 정확도 측정
    model.compile(loss='categorical_crossentropy',
        optimizer='adam', metrics=['accuracy'])

# 연쇄 방식 모델링을 포함하는 함수형 구현
# ANN 모델을 연쇄 방식으로 구현한다. 모델 구현은 함수형 방식을 사용한다.
# 간편하게 계층을 정의 할 수 있다.
# 복잡한 네트워크를 구현해야 할 경우 연쇄형 모델링 만으로 구현이 어려울 수 있다. 이럴때는 분산 방식 모델링 사용
def ANN_seq_func(Nin, Nh, Nout):
    # 모델 생성
    model = models.Sequential()

    # 모델 구조 설정
    # 입력 계층과 은닉 계층 형태가 동시에 설정
    model.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
    # 출력 계층 설정
    model.add(layers.Dense(Nout, activation='softmax'))


# 분산 방식 모델링을 포함하는 객체지향 구현
class ANN_models_class(models.Model):
    def __init__(self, Nin, Nh, Nout):
        hidden = layers.Dense(Nh)
        output = layers.Dense(Nout)
        relu = layers.Activation('relu')
        softmax = layers.Activation('softmax')

        x = layers.Input(shape=(Nin,))
        h = relu(hidden(x))
        y = softmax(output(h))
        
        super().__init(x, y)

        self.compile(loss='categorical_crossentropy',
            optimizer='adam', metrics=['accuracy'])

# 연쇄 방식 모델링을 포함하는 객체지향 구현
class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
            super().__init__()
            self.add(layers.Dense(Nh, activation='relu', input_shape=(Nin,)))
            self.add(layers.Dense(Nout, activation='softmax'))
            self.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy'])


# 학습 training 데이터 - 모델을 학습하는데 사용하는 데이터
# 검증 validation 데이터 - 학습이 진행되는 동안 성능을 검증하는데 사용하는 데이터
# 평가 test 데이터 - 학습을 마치고 나서 모델의 성능을 최종적으로 평가하는데 사용하는 데이터
def Data_func():
    (X_train, y_train), (X_test, y_test) = datasets.mnist.load_data()
    Y_train = np_utils.to_categorical(y_train)
    Y_test = np_utils.to_categorical(y_test)

    L, W, H = X_train.shape
    X_train = X_train.reshape(-1, W * H)
    X_test = X_test.reshape(-1, W * H)

    X_train = X_train / 255.0
    X_test = X_test / 255.0

    return (X_train, Y_train), (X_test, Y_test)

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def plot_acc(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

def main():
    Nin = 784
    Nh = 100
    number_of_class = 10
    Nout = number_of_class
    
    model = ANN_seq_class(Nin, Nh, Nout)
    
    (X_train, Y_train), (X_test, Y_test) = Data_func()
    history = model.fit(X_train, Y_train, epochs=5, batch_size=100, validation_split=0.2)

    performance_test = model.evaluate(X_test, Y_test, batch_size=100)
    print('Test Loss and Accuracy -> {:.2f}, {:.2f}'.format(*performance_test))

    plot_loss(history)
    plt.show()

    plot_acc(history)
    plt.show()

if __name__ == '__main__':
    main()