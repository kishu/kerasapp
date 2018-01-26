"""
mnist for CNN

1. 분류 CNN 모델링
2. 분류 CNN을 위한 데이터 준비
3. 학습 효과 분석
4. 분류 CNN 학습 및 성능 평가
"""

import os
import keras
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class CNN(keras.models.Sequential):
    """
    Conv2DL 2차원 합성곱을 계산하는 클래스
    MaxPooling2D: 2차원 맥스풀링을 계산하는 클래스
    Flatten: 다차원 입ㅁ력을 1차원 입력으로 변환하는 클래스
    backend: 딥러닝 엔진의 함수를 직접 호출하거나 주요 파라미터를 제어할 수 있다.
            주로 케라스가 제공하지 않는 새로운 함수를 만들 경우 사용한다.
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()

        # 1st layers: 32 convolution of 3 x 3 kernel
        # padding: One of "valid" or "same" (case-insensitive).
        #   - "valid" means "no padding".
        #   - "same" results in padding the input such that the output has the same length as the original input.
        self.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

        # 2nd layers: 64 convolution of 3 x 3 kernel
        self.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

        # 인접한 2x2 셀 중에 가장 큰 값(MaxPooling)만 내보내는 부속계층
        self.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

        # 전체 노드 중 25%를 끄는 드롭아웃 수행
        self.add(keras.layers.Dropout(0.25))

        # 2차원 이미지를 1차원 벡터로 변환
        self.add(keras.layers.Flatten())

        self.add(keras.layers.Dense(128, activation='relu'))
        self.add(keras.layers.Dropout(0.5))
        self.add(keras.layers.Dense(num_classes, activation='softmax'))

        self.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])

class Data:
    """ data class"""
    def __init__(self):
        num_classes = 10

        (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

        img_rows, img_cols = X_train.shape[1:]

        if keras.backend.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')

        X_train = X_train / 255
        X_test = X_test / 255

        y_train = keras.utils.to_categorical(y_train)
        y_test = keras.utils.to_categorical(y_test)

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])

    plt.title('Model Loss / accuracy')
    plt.ylabel(['Loss', 'Accuracy'])
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc=0)

    plt.show()

def main():
    data = Data()
    model = CNN(data.input_shape, data.num_classes)

    history = model.fit(data.X_train,
        data.y_train,
        batch_size = 128,
        epochs = 10,
        validation_split=0.2)

    score = model.evaluate(data.X_test, data.y_test)
    print('\nTest Loss and Accuracy ->', score)

    plot_history(history)

if __name__ == '__main__':
    main()
