# ================================================================
# This file contains some useful tools for a classification 
# project at IMT Atlantique. Authors: Martina BALBI, Mateo BENTURA, 
# Ezequiel CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(y_test, y_pred, model_name, dataset_name):
    '''
    Function to plot and save a confusion matrix

    INPUT
    y_test: test labels
    y_train: predicted labels
    model_name: name of the classifier used to predict the labels
    dataset_name: name of the dataset

    AUTHOR
    Martina Balbi
    '''
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    heatmap = sns.heatmap(cm,cmap="YlGnBu",linewidths=.5, annot=True, annot_kws={"size": 18}, fmt="d", xticklabels=['Authentic', 'Fake'], yticklabels=['Authentic', 'Fake'])
    heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=13) 
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=13) 
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion matrix for %s dataset using %s' % (dataset_name, model_name), fontsize=15)
    plt.savefig("plots/confusion_matrixes/Confustion_Matrix_%s_%s.jpg" % ('banknote-auth', 'knn'))
    