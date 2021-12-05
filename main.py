# ================================================================
# This is the main file of a classification project made at IMT 
# Atlantique. Authors: Martina BALBI, Mateo BENTURA, Ezequiel 
# CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from mlxtend.plotting import plot_pca_correlation_graph
from clean_normalize import *
from ml_functions import *

ba = pd.read_csv("data_banknote_authentication.txt")
kd = pd.read_csv("kidney_disease.csv").set_index('id')