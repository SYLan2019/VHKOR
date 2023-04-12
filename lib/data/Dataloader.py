import pandas as pd
from tensorflow.python.keras.utils import np_utils

nb_classes = 63
batch_size = 8


def readucr(filename):
    data = pd.read_csv(filename, header=None, encoding='utf-8')
    Y = data[[0]].values
    data.drop(data.columns[0], axis=1, inplace=True)
    X = data.values
    return X, Y


# load data
x_train_h, y_train = readucr('lib/data/h_train.csv')
x_test_h, y_test = readucr('lib/data/h_test.csv')
x_train_k, y_train_k = readucr('lib/data/k_train.csv')
x_test_k, y_test_k = readucr('lib/data/k_test.csv')
x_train_v, y_train_v = readucr('lib/data/v_train.csv')
x_test_v, y_test_v = readucr('lib/data/v_test.csv')


# check tag
if (y_train == y_train_k).all():
    if(y_train == y_train_v).all():
        print('train tag ok')
    else:
        raise Exception('y_train_h != y_train_v')
else:
    raise Exception('y_train_h != y_train_k')

if (y_test == y_test_k).all():
    if(y_test == y_test_v).all():
        print('test tag ok')
    else:
        raise Exception('y_test_h != y_test_v')
else:
    raise Exception('y_test_h != y_test_k')


def Y_pro(Y_data):
    y = (Y_data - Y_data.min()) / (Y_data.max() - Y_data.min()) * (nb_classes - 1)
    y = np_utils.to_categorical(y, nb_classes)
    return y
Y_train = Y_pro(y_train)
Y_test = Y_pro(y_test)
# y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)
# y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

def X_pro(X_data):
    x_mean = X_data.mean()
    x_std = X_data.std()
    x = (X_data - x_mean) / (x_std)
    x = X_data.reshape(X_data.shape + (1, ))
    return x

# Standardise haptic, kinesthetic data
X1_train = X_pro(x_train_h)
X1_test = X_pro(x_test_h)
X2_train = X_pro(x_train_k)
X2_test = X_pro(x_test_k)
# visual
X3_train = x_train_v.reshape(x_train_v.shape + (1, ))
X3_test = x_test_v.reshape(x_test_v.shape + (1, ))



