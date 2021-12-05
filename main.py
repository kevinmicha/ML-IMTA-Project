# ================================================================
# This is the main file of a classification project made at IMT 
# Atlantique. Authors: Martina BALBI, Mateo BENTURA, Ezequiel 
# CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
from clean_normalize import *
from ml_functions import *
from NN_classifier import *
from sklearn.model_selection import train_test_split

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

# Split dataset into Train and Test
X_train, X_test, y_train, y_test = train_test_split(data_set_df, target_df, test_size=0.3, random_state=48)

y_pred = fit_NN(X_train, X_test, y_train, y_test, 'banknote-auth')