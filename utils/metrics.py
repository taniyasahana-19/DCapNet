import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def iou_score(y_true, y_pred):
    inter = np.logical_and(y_true, y_pred).sum()
    uni = np.logical_or(y_true, y_pred).sum()
    return inter / uni if uni>0 else 0.0

def dice_score(y_true, y_pred):
    inter = np.logical_and(y_true, y_pred).sum()
    denom = y_true.sum() + y_pred.sum()
    return (2*inter)/denom if denom>0 else 0.0

def classification_metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
