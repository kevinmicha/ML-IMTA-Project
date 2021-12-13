# ================================================================
# This file contains some Machine Learning functions used for
# a classification project at IMT Atlantique. Authors: Martina
# BALBI, Mateo BENTURA, Ezequiel CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
from numpy.testing._private.utils import KnownFailureException
import pandas as pd
import matplotlib.pyplot as plt

from lib.nn_util import *

from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_pca_correlation_graph
from sklearn.mixture import GaussianMixture
from sklearn import svm
from sklearn.metrics import accuracy_score


def pca(dataset, dataset_name):
    '''
    Principal Component Analysis (PCA) function

    INPUT
    dataset: A pandas DataFrame after the cleaning step
    dataset_name: A string containing the dataset name

    OUTPUT
    dataset_after_pca: A PCA-transformed version of the original dataset

    AUTHOR
    Kevin Michalewicz
    '''
    ncomp_kd = 10  # number of components for kidney disease
    ncomp_ba = 2  # number of components for banknote authentication

    if dataset_name == 'kidney-disease':
        numerical_columns = dataset.columns[dataset.dtypes == 'float64']
        categorical_columns = dataset.columns[dataset.dtypes == 'uint8']
        pca = PCA(n_components=ncomp_kd)
        pca.fit(dataset[numerical_columns])
        print('{} components represent {:.2f} of the variance'.format(
            ncomp_kd, sum(pca.explained_variance_ratio_)))
        print('---------------------------------')
        # figure, correlation_matrix = plot_pca_correlation_graph(dataset[numerical_columns], numerical_columns, figure_axis_size=10)
        kd_tf_numerical = pca.transform(dataset[numerical_columns])
        dataset_after_pca = pd.concat([pd.DataFrame(
            data=kd_tf_numerical, index=dataset.index), dataset[categorical_columns]], axis=1)

    elif dataset_name == 'banknote-auth':
        pca = PCA(n_components=ncomp_ba)
        pca.fit(dataset)
        print('{} components represent {:.2f} of the variance'.format(
            ncomp_ba, sum(pca.explained_variance_ratio_)))
        print('---------------------------------')
        # figure, correlation_matrix = plot_pca_correlation_graph(dataset, dataset.columns, figure_axis_size=10)
        dataset_after_pca = pd.DataFrame(
            pca.transform(dataset), index=dataset.index)

    return dataset_after_pca


def fit_nn(X_train, X_test, y_train, y_test, dataset_name):
    '''
    Neural Network Classifier

    INPUT
    X_train: features for training
    X_test: features for testing
    y_train: targets for training
    y_test: targets for testing
    dataset_name: A string containing the dataset name

    OUTPUT
    y_predicted: predicted labels for the testing set

    AUTHOR
    Ezequiel Centofanti
    '''
    if dataset_name == 'kidney-disease':
        nb_features = 20

    elif dataset_name == 'banknote-auth':
        nb_features = 4

    # Create data-loaders
    train_loader, val_loader, test_loader = create_torch_dataset(
        X_train, X_test, y_train, y_test)

    # Initialize the neural network
    model_ = Net1(nb_features)

    # Specify loss function (categorical cross-entropy)
    criterion = nn.BCELoss()

    # Specify optimizer (stochastic gradient descent) and learning rate
    optimizer = torch.optim.SGD(model_.parameters(), lr=0.05)

    # Train model
    n_epochs = 80  # number of epochs to train the model
    train_losses_1, valid_losses_1 = training(
        n_epochs,
        train_loader,
        val_loader,
        model_,
        criterion,
        optimizer)

    # Plot loss over training
    plot_losses(train_losses_1, valid_losses_1, n_epochs, dataset_name)
    
    return evaluation(model_, test_loader, criterion, dataset_name)


def fit_knn(X_train, X_test, y_train, y_test, dataset_name):
    '''
    K-Nearest Neighbors classifier

    INPUT
    X_train: features for training
    X_test: features for testing
    y_train: targets for training
    y_test: targets for testing
    dataset_name: A string containing the dataset name

    OUTPUT
    y_predicted: predicted labels for the testing set

    AUTHOR
    Martina Balbi
    '''
    K = 5

    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(X_train, y_train)
    y_predicted = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)

    print('K-Nearest neighbors test accuracy for dataset %s: %.2f (%2d/%2d)' %
          (dataset_name, accuracy * 100, accuracy * len(y_test), len(y_test)))
    print('---------------------------------')

    return y_predicted


def fit_gmm(X_train, X_test, y_train, y_test, dataset_name):
    '''
    Gaussian mixture model classifier

    INPUT
    X_train: features for training
    X_test: features for testing
    y_train: targets for training (**not used: unsupervised method**)
    y_test: targets for testing
    dataset_name: A string containing the dataset name

    OUTPUT
    y_predicted: predicted labels for the testing set

    AUTHOR
    Mateo Bentura
    '''
    gm = GaussianMixture(n_components=y_train.nunique())
    gm.fit(X_train)
    y_predicted = gm.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print('Gaussian mixture model test for dataset %s: %2d%% (%2d/%2d)' %
          (dataset_name, accuracy * 100, accuracy * len(y_test), len(y_test)))
    print('---------------------------------')
    # X_samples, y_samples = gm.sample(n_samples=1000)

    return y_predicted, gm.means_, gm.covariances_

def fit_svm(X_train, X_test, y_train, y_test, dataset_name):
    '''
    Support Vector Machine (SVM) classifier

    INPUT
    X_train: features for training
    X_test: features for testing
    y_train: targets for training (**not used: unsupervised method**)
    y_test: targets for testing
    dataset_name: A string containing the dataset name

    OUTPUT
    y_predicted: predicted labels for the testing set

    AUTHOR
    Kevin Michalewicz
    '''
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_predicted)
    print('Support Vector Machine test for dataset %s: %2d%% (%2d/%2d)' %
          (dataset_name, accuracy * 100, accuracy * len(y_test), len(y_test)))
    print('---------------------------------')

    return y_predicted
