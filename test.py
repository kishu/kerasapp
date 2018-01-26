
# https://github.com/jskDr/keraspp
from keras import datasets
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

X_train = X_train.astype('float32')

print('')
print('X_train shape', X_train.shape) # (50000, 32, 32, 3)

print('')
print('X_train.shape[1:]', X_train.shape[1:])

print('')
print('X_train.reshape(50000, -1)')
X_train_reshape = X_train.reshape(50000, -1) # (50000, 3072)

print('')
print('X_train_reshape.shape', X_train_reshape.shape)
print('X_train_reshape[0]', X_train_reshape[0])

# (50000, 32, 32, 3)
scaler = MinMaxScaler()
X_train_transform = scaler.fit_transform(X_train_reshape)
X_train = X_train_transform.reshape(X_train.shape)

print('')
print('scaler.fit_transform(X_train_reshape)')
print('X_train_transform[0]', X_train_transform[0])

# print('\nX_train shape', X_train.shape)
