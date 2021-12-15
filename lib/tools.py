# ================================================================
# This file contains some useful tools for a classification 
# project at IMT Atlantique. Authors: Martina BALBI, Mateo BENTURA, 
# Ezequiel CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np

def plot_confusion_matrix(y_test, y_pred, model_name, dataset_name):
    '''
    Function to plot and save a confusion matrix

    INPUT
    y_test: test labels
    y_pred: predicted labels
    model_name: name of the classifier used to predict the labels
    dataset_name: name of the dataset

    AUTHOR
    Martina Balbi
    '''
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,8))
    heatmap = sns.heatmap(cm,cmap="YlGnBu",linewidths=.5, annot=True, annot_kws={"size": 18}, fmt="d")
    if dataset_name == 'banknote-authentication':
        heatmap.set_xticklabels(['Authentic', 'Fake'], fontsize=13) 
        heatmap.set_yticklabels(['Authentic', 'Fake'], fontsize=13) 
    elif dataset_name == 'kidney-disease':
        heatmap.set_xticklabels(['Not Diseased', 'Diseased'], fontsize=13) 
        heatmap.set_yticklabels(['Not Diseased', 'Diseased'], fontsize=13)
    plt.xlabel('Predicted', fontsize=14)
    plt.ylabel('True', fontsize=14)
    plt.title('Confusion matrix for %s dataset using %s' % (dataset_name, model_name), fontsize=15)
    plt.savefig("plots/confusion_matrices/Confusion_Matrix_%s_%s.jpg" % (dataset_name, model_name))

def get_scores(y_test, y_pred, model_name, dataset_name):
    '''
    Function to calculate scores of the model (precision, recall, F1-score)

    INPUT
    y_test: test labels
    y_pred: predicted labels
    model_name: name of the classifier used to predict the labels
    dataset_name: name of the dataset

    AUTHOR
    Martina Balbi
    '''

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    score = f1_score(y_test, y_pred)
    print('%s F1 Score for dataset %s: %.2f, precision: %.2f, recall: %.2f' %
          (model_name, dataset_name, score, precision, recall))
    print('---------------------------------')

    return 