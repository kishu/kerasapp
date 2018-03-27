import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras import datasets
from sklearn import preprocessing

(X_train, y_train), (X_test, y_test) = datasets.boston_housing.load_data()

# Transforms features by scaling each feature to a given range.
#  feature_range : tuple (min, max), default=(0, 1)
#  Desired range of transformed data.
scaler = preprocessing.MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(X_train[0])
print(y_train[0])

model = Sequential()
model.add(Dense(1, input_dim=13, activation='linear'))
model.compile(loss='mse', optimizer='sgd')

result = model.fit(X_train, y_train, epochs=1000)

#plt.plot(result.history['loss'])
#plt.show()

X_pred = np.array([[0.00632, 18.00, 2.310, 0., 0.5380, 6.5750, 65.20, 4.0900, 1., 296.0, 15.30, 396.90, 4.98]])
X_pred = scaler.fit_transform(X_pred)

predict = model.predict(X_pred)

print("predict", predict)

#performance = model.evaluate(X_test, y_test)
#print('\nTest Loss -> {:.2f}'.format(performance))
