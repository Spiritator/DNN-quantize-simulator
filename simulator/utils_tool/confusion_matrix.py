"""
reference: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

================
Confusion matrix
================

Example of confusion matrix usage to evaluate the quality
of the output of a classifier on the iris data set. The
diagonal elements represent the number of points for which
the predicted label is equal to the true label, while
off-diagonal elements are those that are mislabeled by the
classifier. The higher the diagonal values of the confusion
matrix the better, indicating many correct predictions.

The figures show the confusion matrix with and without
normalization by class support size (number of elements
in each class). This kind of normalization can be
interesting in case of class imbalance to have a more
visual interpretation of which class is being misclassified.

Here the results are not as good as they could be as our
choice for the regularization parameter C was not the best.
In real life applications this parameter is usually chosen
using :ref:`grid_search`.

"""

#print(__doc__)

import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

def _plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          big_matrix=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        #print('Confusion matrix, without normalization')
        pass

    #print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    if not big_matrix:
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if not big_matrix:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def show_confusion_matrix(y_test,y_pred,class_names,title,figsize=None,normalize=False,big_matrix=False,dpi=None):
    """
    Show confusion matrix of DNN inference result

    Parameters
    ----------
    y_test : Ndarray
        The label of test dataset.
    y_pred : Ndarray
        The DNN prediction of test dataset.
    class_names : List of String
        The name of each dataset class.
    title : String
        The confusion matrix figure title.
    figsize : Tuple of Integer, optional. (width, height)
        The size for confusion matrix figure, the unit is inch. The default is None.
    normalize : Bool, optional
        Scale the matrix value to between 0 and 1 instead of showing number count. The default is False.
    big_matrix : Bool, optional
        If True, don't show the class names which prevents chaotic layout of figure. The default is False.
    dpi : Integer, optional
        Definition per inch, the image quality. The default is None.

    Returns
    -------
    None
        Show the confusion matrix in terminal.

    """
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)
    
    plt.figure(figsize=figsize,dpi=dpi)
    _plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=normalize,title=title,big_matrix=big_matrix)
    
    plt.show()
