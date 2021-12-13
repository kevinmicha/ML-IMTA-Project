# ================================================================
# This file contains some useful tools for a classification 
# project at IMT Atlantique. Authors: Martina BALBI, Mateo BENTURA, 
# Ezequiel CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import numpy as np

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
    


def plot_gmm_covariances(X_train, y_train, means, covariances):
    '''
    Function to plot covariances of the gaussian mixture model.

    INPUT
    y_test: test labels
    y_train: predicted labels
    model_name: name of the classifier used to predict the labels
    dataset_name: name of the dataset

    AUTHOR
    Mateo Bentura
    '''
    plt.figure(figsize=(8,8))
    ax = plt.gca()
    # plt.scatter(X_samples_gmm_ba[:,0], X_samples_gmm_ba[:,1], c=y_samples_gmm_ba, cmap='GnBu')
    colors = ["tab:red", "tab:blue"]
    # colors = ["navy", "darkorange"]
    for n in range(2):
        ax.scatter(X_train[y_train==n][0], X_train[y_train==n][1], c=colors[n], label=str(n), alpha=0.6, marker='x')
        eig_val, eig_vect = np.linalg.eigh(covariances[n])
        radius_x, radius_y = np.sqrt(2)*np.sqrt(eig_val)
        u = eig_vect[0] / np.linalg.norm(eig_vect[0])
        angle = 180 * np.arctan2(u[1], u[0]) / np.pi + 180
        ell = patches.Ellipse(
            means[n], width=radius_x*2, height=radius_y*2, angle=angle, color=colors[n]
        )
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)
        ax.set_aspect("equal", "datalim")

    plt.legend()
    plt.title('Scatter plot of PCA components for dataset banknote-authentication\nwith Gaussian mixture models')
    plt.savefig("plots/covariance_plots/Covariance_Plot_banknote-authentication")