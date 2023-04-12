from lib.model.trans import build_T
from lib.model.crosstrans import build_T_cross
# from LMF import model_LMF
from lib.data.Dataloader import X1_train, X2_train, X3_train, X1_test, X2_test, X3_test, Y_train, Y_test
import matplotlib.pyplot as plt
import pickle
from tensorflow import keras
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import argparse
import os
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn import metrics


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--model', type=str)
args = parser.parse_args()
nb_epochs = args.epochs
batch_size = args.batch_size
model_name = args.model
if model_name == 'SA':
    model1 = build_T()
elif model_name == 'CA':
    model1 = build_T_cross()

optimizer = keras.optimizers.Adam()
model1.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
callbacks = [
    ModelCheckpoint('model_1.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max', period=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.0001),
    # keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='auto'),
    # keras.callbacks.TensorBoard(log_dir='./tb', histogram_freq=1)
]
print('haptic shape:', X1_train.shape, '\n kinesthetics shape:', X2_train.shape, '\n visual shape:', X3_train.shape,  '\n y shape:', Y_train.shape)
history = model1.fit([X1_train, X2_train, X3_train], Y_train, batch_size=batch_size, epochs=nb_epochs, verbose=1, validation_data=([X1_test, X2_test, X3_test], Y_test), callbacks=callbacks)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# save:
with open('model_1.txt', 'wb') as file_txt:
    pickle.dump(history.history, file_txt)

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, color='red', label='Validation acc')
plt.title('visual-haptic-kinesthetics Training accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, color='red', label='Validation loss')
plt.title('visual-haptic-kinesthetics Training loss')
plt.legend()

plt.show()

score = model1.evaluate([X1_test, X2_test, X3_test], Y_test)
print("Accuracy after loading Model:", score[1]*100)


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(15, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()


# Show confusion matrix
def plot_confuse(model, x_val, y_val, labels):
    predictions = model.predict(x_val)
    predictions = predictions.argmax(axis=1)
    truelabel = y_val.argmax(axis=1)  # one-hot to label
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    recall = metrics.recall_score(truelabel, predictions, average='weighted')
    f1_score = metrics.f1_score(truelabel, predictions, average='weighted')
    print("recall:", recall)
    print("f1_score:", f1_score)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix')


labels = [i for i in range(1,64)]
plot_confuse(model1, [X1_test, X2_test, X3_test], Y_test, labels)