"""
LSTM example for Keras

1. 라이브러리 임포트
2. 데이터 준비
3. 모델링
4. 학습 및 성능 평가
"""

import os
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Data:
    def __init__(self, max_features=20000, maxlen=80):

        # 기본적으로 알아야 할 점은 우리가 하려는게 자연어처리라고 해서 character, string을 입력으로 때려박는 건 아니라는 점입니다.
        # 먼저 데이터를 딥러닝 모델에 입력하기 좋은 상태(숫자)로 만들어줘야 합니다.
        # 먼저 데이터에서 등장하는 각각의 단어에 고유의 번호(index)를 붙이고, 문장을 단어 index의 sequence로 만들어야 합니다.
        # 간단히 말해서 "I eat apple." 이라는 문장을 [1, 12, 43]과 같이 바꾸는 것입니다.
        # 이렇게 문장을 sequence로 바꿨다면, 이 번호들을 이용해 단어를 임베딩해서 문장을 "문장 길이 x 임베딩 차원"의 2차원짜리 입력으로 주는 것입니다.

        # x_trian: IMDB의 리뷰
        # x_train.shape: (25000,)
        # 사전 처리 한 단어별 정수형 인덱스
        # 전체 단어 중 사용하는 빈도에 따라 인덱싱.

        # -------------------------------
        # 단어 | 사용빈도 | 사용빈도순위 | 포함
        #--------------------------------
        # a     30000     1           O
        # b     20000     2           O
        # .     ...       .           O
        # x     1000      20000       O
        # y     100       30000       X
        # z     10        40000       X

        # 20000 번째로 많이 사용하는 단어까지만 추출한다.
        # y_train: 리뷰에 대한 평가
        # y_train.shape: (25000,)
        # 1-positive or 0-negative
        (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

        # print(len(x_train[0])) > 218
        # print(len(x_train[1])) > 189

        # 데이터 셋에 들어 있는 문장들은 길이가 다르기 떄문에 LSTM이 처리하기 적합하도록 길이를 통일한다.
        # 문장의 길이가 maxlen보다 작으면 부족한 부분을 0으로 채운다.
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
        x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

        # print(len(x_train[0])) > 80
        # print(len(x_train[1])) > 80

        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test

# https://brunch.co.kr/@chris-song/9
class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):

        x = layers.Input((maxlen,))

        # 단어를 의미론적 기하공간에 핑할 수 있도록 벡터화한다.
        # 어떤 기준으로 벡터화 하는가?????????
        # keras.layers.Embedding(input_dim, output_dim)(input_length)
        # This layer can only be used as the first layer in a model.
        #   input_dim
        #     단어 사전의 크기를 말하며 총 20,000개의 단어 종류가 있다는 의미입니다.
        #     이 값은 imdb.load_data() 함수의 num_words 인자값과 동일해야 합니다.
        #   output_dim
        #     단어를 인코딩한 후 나오는 벡터 크기입니다.
        #     이 값이 128이라면 단어를 128차원의 의미론적 기하공간에 나타낸다는 의미입니다.
        #     단순하게 빈도수만으로 단어를 표시한다면, 10과 11을 빈도수는 비슷하지만 단어로 볼 때는 전혀 다른 의미를 가지고 있습니다.
        #     하지만 의미론적 기하공간에서는 거리가 가까운 두 단어는 의미도 유사합니다.
        #     즉, 임베딩 레이어는 입력되는 단어를 의미론적으로 잘 설계된 공간에 위치시켜 벡터로 수차화 시킨다고 볼 수 있습니다.
        #   input_length
        #     단어의 수 즉 문장의 길이를 나타냅니다. 임베딩 레이어의 출력 크기는 샘플수 output_dim input_length가 됩니다.
        #     임베딩 레이어 다음에 플래튼 레이어가 온다면 반드시 input_length를 지정해야 합니다.
        #     플래튼 레이어인 경우 입력 크기가 알아야 이를 1차원으로 만들어서 Dense 레이어에 전달할 수 있기 때문입니다.
        h = layers.Embedding(max_features, 128)(x)
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
        y = layers.Dense(1, activiation='sigmoid')(h)

        super().__init(x, y)
        self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class RNN_LSTM(models.Model):
    def __init__(self, max_features, maxlen):

        x = layers.Input((maxlen,))

        # 단어를 의미론적 기하공간에 핑할 수 있도록 벡터화한다.
        # 어떤 기준으로 벡터화 하는가?????????
        # keras.layers.Embedding(input_dim, output_dim)(input_length)
        # This layer can only be used as the first layer in a model.
        #   input_dim
        #     단어 사전의 크기를 말하며 총 20,000개의 단어 종류가 있다는 의미입니다.
        #     이 값은 imdb.load_data() 함수의 num_words 인자값과 동일해야 합니다.
        #   output_dim
        #     단어를 인코딩한 후 나오는 벡터 크기입니다.
        #     이 값이 128이라면 단어를 128차원의 의미론적 기하공간에 나타낸다는 의미입니다.
        #     단순하게 빈도수만으로 단어를 표시한다면, 10과 11을 빈도수는 비슷하지만 단어로 볼 때는 전혀 다른 의미를 가지고 있습니다.
        #     하지만 의미론적 기하공간에서는 거리가 가까운 두 단어는 의미도 유사합니다.
        #     즉, 임베딩 레이어는 입력되는 단어를 의미론적으로 잘 설계된 공간에 위치시켜 벡터로 수차화 시킨다고 볼 수 있습니다.
        #   input_length
        #     단어의 수 즉 문장의 길이를 나타냅니다. 임베딩 레이어의 출력 크기는 샘플수 output_dim input_length가 됩니다.
        #     임베딩 레이어 다음에 플래튼 레이어가 온다면 반드시 input_length를 지정해야 합니다.
        #     플래튼 레이어인 경우 입력 크기가 알아야 이를 1차원으로 만들어서 Dense 레이어에 전달할 수 있기 때문입니다.
        h = layers.Embedding(max_features, 128)(x)
        h = layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)(h)
        y = layers.Dense(1, activation='sigmoid')(h)

        super().__init__(x, y)

        self.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

class Machine:
    def __init__(self, max_features=20000, maxlen=80):
        self.data = Data(max_features, maxlen)
        self.model = RNN_LSTM(max_features, maxlen)

    def run(self, epochs=3, batch_size=32):
        data = self.data
        model = self.model

        print('\nTraining stage')
        print('=======================')

        model.fit(data.x_train, data.y_train,
            batch_size=batch_size, epochs=epochs,
            validation_data=(data.x_test, data.y_test))

        print('\nEvaluating stage')
        print('=======================')

        score, acc = model.evaluate(data.x_test, data.y_test, batch_size=batch_size)

        print('Test performance: accuracy={0}, loss={1}'.format(acc, score))

def main():
    m = Machine()
    m.run()

if __name__ == '__main__':
    main()
