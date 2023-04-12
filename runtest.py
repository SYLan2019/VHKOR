import matplotlib.pyplot as plt
import pickle
from tensorflow.keras.models import load_model
from lib.data.Dataloader import X1_test, X2_test, X3_test, Y_test
import itertools
from sklearn.metrics import confusion_matrix
import numpy as np
from lib.model.trans import CustLayer
from sklearn import metrics
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
args = parser.parse_args()
model_name = args.model
if model_name == 'SA':
    model_name = 'lib/save_model/CRNN-SA'
elif model_name == 'CA':
    model_name = 'lib/save_model/CRNN-CA'

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


def plot_confuse(model, x_val, y_val, labels):
    predictions = model.predict(x_val)
    predictions = predictions.argmax(axis=1)
    truelabel = y_val.argmax(axis=1)
    conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
    recall = metrics.recall_score(truelabel, predictions, average='weighted')
    f1_score = metrics.f1_score(truelabel, predictions, average='weighted')
    print("recall:", recall)
    print("f1_score:", f1_score)
    plt.figure()
    plot_confusion_matrix(conf_mat, normalize=False, target_names=labels, title='Confusion Matrix')


def figure():
    # retrieve:
    with open(model_name + '.txt', 'rb') as file_txt:
        history = pickle.load(file_txt)
    acc = history['accuracy']
    val_acc = history['val_accuracy']
    loss = history['loss']
    val_loss = history['val_loss']
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

model = load_model(model_name + '.h5', custom_objects={'CustLayer': CustLayer})
score = model.evaluate([X1_test, X2_test, X3_test], Y_test)
print("Accuracy after loading Model:", score[1]*100)


labels = [i for i in range(1, 64)]

plot_confuse(model, [X1_test, X2_test, X3_test], Y_test, labels)
