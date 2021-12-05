# ================================================================
# This file contains some Machine Learning functions used for 
# a classification project at IMT Atlantique. Authors: Martina 
# BALBI, Mateo BENTURA, Ezequiel CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
import pandas as pd

from NN_util import *

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_pca_correlation_graph

def pca(dataset, dataset_name):
    '''
    Principal Component Analysis (PCA) function

    INPUT
    dataset: A pandas DataFrame after the cleaning step
    dataset_name: A string containing the dataset name

    OUTPUT
    dataset_after_pca: A PCA-transformed version of the original dataset
    
    '''
    ncomp_kd = 10 # number of components for kidney disease
    ncomp_ba = 2 # number of components for banknote authentication

    if dataset_name == 'kidney-disease':
        numerical_columns = dataset.columns[dataset.dtypes=='float64']
        categorical_columns = dataset.columns[dataset.dtypes=='uint8']
        pca = PCA(n_components=ncomp_kd)
        pca.fit(dataset[numerical_columns])
        print('{} components represent {:.2f} of the variance'.format(ncomp_kd, sum(pca.explained_variance_ratio_)))
        # figure, correlation_matrix = plot_pca_correlation_graph(dataset[numerical_columns], numerical_columns, figure_axis_size=10)
        kd_tf_numerical = pca.transform(dataset[numerical_columns])
        dataset_after_pca = pd.concat([pd.DataFrame(data=kd_tf_numerical, index=dataset.index), dataset[categorical_columns]], axis=1)

    elif dataset_name == 'banknote-auth':
        pca = PCA(n_components=ncomp_ba)
        pca.fit(dataset)
        print('{} components represent {:.2f} of the variance'.format(ncomp_ba, sum(pca.explained_variance_ratio_)))
        # figure, correlation_matrix = plot_pca_correlation_graph(dataset, dataset.columns, figure_axis_size=10)
        dataset_after_pca = pd.DataFrame(pca.transform(dataset), index=dataset.index)

    return dataset_after_pca

def fit_NN(X_train, X_test, y_train, y_test, dataset_name):
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

    #test_loader, train_loader = create_torch_dataset(data_set_df, target_df)
    train_loader, test_loader= create_torch_dataset(X_train, X_test, y_train, y_test)

    # initialize the neural network
    model_ = Net1(nb_features) 

    # Specify loss function (categorical cross-entropy)
    criterion = nn.BCELoss()

    # Specify optimizer (stochastic gradient descent) and learning rate
    optimizer = torch.optim.SGD(model_.parameters(),lr = 0.05) 

    # Train model
    n_epochs = 80 # number of epochs to train the model
    train_losses_1 = training(n_epochs, train_loader, model_, criterion, optimizer)

    # Plot loss over training
    # plt.plot(range(n_epochs), train_losses_1)
    # plt.legend(['train', 'validation'], prop={'size': 10})
    # plt.title('loss function', size=10)
    # plt.xlabel('epoch', size=10)
    # plt.ylabel('loss value', size=10)

    return evaluation(model_, test_loader, criterion)