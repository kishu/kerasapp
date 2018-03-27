import os
import matplotlib.pyplot as plt
from keras import layers, models
from keras.datasets import mnist

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 파이썬에서는 보통 사용하지 않는 변수는 _로 받는다.
(X_train, _), (X_test, _) = mnist.load_data()

# print(X_train.shape) #(60000, 28, 28)
# print(X_test.shape) #(10000, 28, 28)

# 각 픽셀 값을 1이하로 정규화 한다.
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

# DNN 즉 완전 연결 계층 구조에 적합하도록 3차원에서 2차원으로 축소
# print(X_train.shape) #(60000, 784)
# print(X_test.shape) #(10000, 784)

# encoder
# 입력계층과 출력 계층의 노드 수를 같게
x_node = X_train.shape[1] # 784
h_unit = 36

x = layers.Input(shape=(x_node,))
h = layers.Dense(h_unit, activation='relu')(x)
y = layers.Dense(x_node, activation='sigmoid')(h)

m = models.Model(x, y)
m.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

# 입력과 출력을 모두 X_train으로 설정하여 학습.
# 학습에 들어가는 데이터는 무작위 순서로 섞어 들어갑니다.
history = m.fit(X_train,
    X_train,
    epochs=10,
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test))


# encoder
encoder = models.Model(x, h)

# decoder
di = layers.Input(shape=(h_unit,))
dy = (m.layers[-1])(di) # 제일 마지막 부분 즉 AE 자신의 출력 계층

decoder = models.Model(di, dy)


encoded_imgs = encoder.predict(X_test)
print(encoded_imgs.shape)
decoded_imgs = decoder.predict(encoded_imgs)

n = 10
plt.figure(figsize=(20,6))

for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(X_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n)
    plt.stem(encoded_imgs[i].reshape(-1))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
