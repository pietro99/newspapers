from sklearn.metrics import confusion_matrix
import numpy as np
def accuracy_per_class(targets, predicted_class, labels=None):
    '''
    given target classes and predicted classes returns the accuracy per class
    Arguments:
        targets (array of int): integers representing the targets classes
        predicted_class (array of int): integers representing the predicted classes
        labels (array of int): labels to consider 
    Returs:
        accuracies (array of int): accuracy for each class
    '''
    cm = confusion_matrix(targets, predicted_class, labels=labels).astype(np.float64)
    numerator = cm.diagonal()
    denominator = cm.sum(axis=1)
    accuracies = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
    accuracies = np.where(denominator == 0, np.nan, accuracies)
    return accuracies
