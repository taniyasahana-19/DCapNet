import matplotlib.pyplot as plt
import numpy as np
import os

def plot_confusion_matrix(cm, classes, out_path=None):
    plt.figure(figsize=(8,6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', color='white' if cm[i,j] > thresh else 'black')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path)
    else:
        plt.show()
    plt.close()

def plot_training_curves(history, out_dir=None, name='curve'):
    plt.figure(figsize=(12,5))
    if 'train_loss' in history and 'val_loss' in history:
        plt.subplot(1,2,1)
        plt.plot(history['train_loss'], label='train')
        plt.plot(history['val_loss'], label='val')
        plt.title('Loss')
        plt.legend()
    if 'train_acc' in history and 'val_acc' in history:
        plt.subplot(1,2,2)
        plt.plot(history['train_acc'], label='train')
        plt.plot(history['val_acc'], label='val')
        plt.title('Accuracy')
        plt.legend()
    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, f'{name}.png'))
    else:
        plt.show()
    plt.close()
