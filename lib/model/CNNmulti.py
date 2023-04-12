from __future__ import print_function
from lib.data.Dataloader import X1_train, X2_train, X3_train
from tensorflow import keras
from tensorflow.keras.models import Model


'''x1-out1:model_haptic'''
def CNN_H():
    x1 = keras.layers.Input(X1_train.shape[1:], name='Input_1')
    # drop_out = Dropout(0.2)(x1)
    conv1 = keras.layers.Conv1D(128, 8, padding="same")(x1)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation('relu')(conv1)

    # drop_out = Dropout(0.2)(conv1)
    conv2 = keras.layers.Conv1D(256, 5, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    # drop_out = Dropout(0.2)(conv2)
    conv3 = keras.layers.Conv1D(128, 3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    pool1 = keras.layers.AveragePooling1D(pool_size=2)(conv3)
    rnn1 = keras.layers.Bidirectional(keras.layers.LSTM(128), merge_mode='sum')(pool1)
    # full1 = keras.layers.GlobalAveragePooling1D()(rnn1)
    # full1 = keras.layers.MaxPooling1D()(rnn1)
    # out1 = keras.layers.Dense(nb_classes, activation='softmax')(full1)
    out1 = keras.layers.Reshape((128, 1))(rnn1)
    cnn1 = Model(x1, out1)
    return cnn1


'''x2-out2:model_kinesthetics'''
def CNN_K():
    x2 = keras.layers.Input(X2_train.shape[1:], name='Input_2')
    # drop_out = Dropout(0.2)(x2)
    conv4 = keras.layers.Conv1D(128, 8, padding="same")(x2)
    conv4 = keras.layers.BatchNormalization()(conv4)
    conv4 = keras.layers.Activation('relu')(conv4)

    # drop_out = Dropout(0.2)(conv4)
    conv5 = keras.layers.Conv1D(256, 5, padding="same")(conv4)
    conv5 = keras.layers.BatchNormalization()(conv5)
    conv5 = keras.layers.Activation('relu')(conv5)

    # drop_out = Dropout(0.2)(conv5)
    conv6 = keras.layers.Conv1D(128, 3, padding="same")(conv5)
    conv6 = keras.layers.BatchNormalization()(conv6)
    conv6 = keras.layers.Activation('relu')(conv6)

    full2 = keras.layers.GlobalAveragePooling1D()(conv6)
    # out2 = keras.layers.Dense(nb_classes, activation='softmax')(full2)
    out2 = keras.layers.Reshape((128, 1))(full2)
    cnn2 = Model(x2, out2)
    return cnn2


'''x3-out3:model_visual'''
def CNN_V():
    x3 = keras.layers.Input(X3_train.shape[1:], name='Input_3')
    # drop_out = Dropout(0.2)(x3)
    conv7 = keras.layers.Conv1D(128, 8, padding="same")(x3)
    conv7 = keras.layers.BatchNormalization()(conv7)
    conv7 = keras.layers.Activation('relu')(conv7)

    # drop_out = Dropout(0.2)(conv7)
    conv8 = keras.layers.Conv1D(256, 5, padding="same")(conv7)
    conv8 = keras.layers.BatchNormalization()(conv8)
    conv8 = keras.layers.Activation('relu')(conv8)

    # drop_out = Dropout(0.2)(conv8)
    conv9 = keras.layers.Conv1D(128, 3, padding="same")(conv8)
    conv9 = keras.layers.BatchNormalization()(conv9)
    conv9 = keras.layers.Activation('relu')(conv9)

    full3 = keras.layers.GlobalAveragePooling1D()(conv9)
    # out3 = keras.layers.Dense(nb_classes, activation='softmax')(full3)
    out3 = keras.layers.Reshape((128, 1))(full3)
    cnn3 = Model(x3, out3)
    return cnn3
