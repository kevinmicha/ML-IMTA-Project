# ================================================================
# This file contains some Machine Learning functions used for 
# a classification project at IMT Atlantique. Authors: Martina 
# BALBI, Mateo BENTURA, Ezequiel CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================

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