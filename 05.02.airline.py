#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection # 데이터를 학습과 검증용으로 나누는 함수
from keras import models, layers

#%% load data
df = pd.read_csv(
    'international-airline-passengers.csv',
    usecols=[1],
    engine='python',
    skipfooter=3)

print(df.head())
print(df.shape)
print(df.values)

passengers = df.values.reshape(-1)

print(passengers.shape)
print(passengers)

#%% plot
plt.xlabel('Time')
plt.ylabel('#Passengers')
plt.title('Original Data')
plt.plot(passengers)

#%% normalize
passengers = \
    (passengers - np.mean(passengers)) / np.std(passengers) / 5

plt.xlabel('Time')
plt.ylabel('#Passengers')
plt.title('Original Data')
plt.plot(passengers)

#%%
M = 4 # How many month to make vactor

X_l = []
y_l = []

# print(passengers)

for ii in range(len(passengers) - M - 1):
    # print(ii, passengers[ii:ii + M], passengers[ii + M])
    X_l.append(passengers[ii:ii + M])
    y_l.append(passengers[ii + M])

#print(X)
X = np.array(X_l)
X = X.reshape(X.shape[0], X.shape[1], 1)
#print(X)
y = np.array(y_l)

X_train, X_test, y_train, y_test = \
    model_selection.train_test_split(X, y, test_size=0.2, random_state=42)


#%%
# print(X_train.shape[1:])
x = layers.Input(shape=X_train.shape[1:])
h = layers.LSTM(10)(x)
y = layers.Dense(1)(h)

m = models.Model(x, y)
m.compile('adam', 'mean_squared_error')

m.summary()

print("fitting...")
history = m.fit(
    X_train,
    y_train,
    epochs=400,
    validation_data=[X_test, y_test])

#%%
plt.title('Model Loss')
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Test'], loc=0)

plt.show()
