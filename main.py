# ================================================================
# This is the main file of a classification project made at IMT 
# Atlantique. Authors: Martina BALBI, Mateo BENTURA, Ezequiel 
# CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
from lib.clean_normalize import *
from lib.ml_functions import *
from lib.nn_util import *
from sklearn.model_selection import train_test_split
from lib.tools import *

# Import and clean datasets
ba = pd.read_csv("datasets/data_banknote_authentication.txt")
kd = pd.read_csv("datasets/kidney_disease.csv").set_index('id')
kd, y_kd = clean_normalize_kd(kd)
ba, y_ba = clean_normalize_ba(ba)

# Perform PCA for the kidney-disease dataset
kd = pca(kd, 'kidney-disease')

# Select dataset to fit
data_set_df = ba
target_df = y_ba

# Split datasets into Train and Test
X_train_ba, X_test_ba, y_train_ba, y_test_ba = train_test_split(ba, y_ba, test_size=0.3, random_state=48)
X_train_kd, X_test_kd, y_train_kd, y_test_kd = train_test_split(kd, y_kd, test_size=0.3, random_state=48)

#K Nearest neighbors
y_pred_knn_ba = fit_knn(X_train_ba, X_test_ba, y_train_ba, y_test_ba, 'banknote-auth')
y_pred_knn_kd = fit_knn(X_train_kd, X_test_kd, y_train_kd, y_test_kd, 'kidney-disease')

#Support Vector Machines
#y_pred_svm_ba = fit_svm(X_train_ba, X_test_ba, y_train_ba, y_test_ba, 'banknote-auth')
#y_pred_svm_kd = fit_svm(X_train_kd, X_test_kd, y_train_kd, y_test_kd, 'kidney-disease')

#Support Vector Machines
#y_pred_gmm_ba = fit_gmm(X_train_ba, X_test_ba, y_train_ba, y_test_ba, 'banknote-auth')
#y_pred_gmm_kd = fit_gmm(X_train_kd, X_test_kd, y_train_kd, y_test_kd, 'kidney-disease')

# Neural network
y_pred_nn_ba = fit_nn(X_train_ba, X_test_ba, y_train_ba, y_test_ba, 'banknote-auth')
y_pred_nn_kd = fit_nn(X_train_kd, X_test_kd, y_train_kd, y_test_kd, 'kidney-disease')


# Plot confusion matrixes
models = ['knn', 'svm', 'gmm', 'nn']
#y_pred = [[y_pred_knn_ba, y_pred_knn_kd], [y_pred_svm_ba, y_pred_svm_kd], [y_pred_gmm_ba, y_pred_gmm_kd], [y_pred_nn_ba, y_pred_nn_kd]]
#y_test = [y_test_ba, y_test_kd]
datasets = ['banknote-authentication', 'kidney-disease']


#mientras que no estan todos los modelos (BORRAR DESPUES)
plot_confusion_matrix(y_test_ba, y_pred_knn_ba, models[0], datasets[0])
plot_confusion_matrix(y_test_kd, y_pred_knn_kd, models[0], datasets[1])

#for i in range(len(models)):
#    for j in range(len(y_test)):
#        plot_confusion_matrix(y_test[j], y_pred[i][j], models[i], datasets[j])