# ================================================================
# This is the main file of a classification project made at IMT 
# Atlantique. Authors: Martina BALBI, Mateo BENTURA, Ezequiel 
# CENTOFANTI and Kevin MICHALEWICZ.
# ================================================================
from clean_normalize import *
from ml_functions import *

ba = pd.read_csv("datasets/data_banknote_authentication.txt")
kd = pd.read_csv("datasets/kidney_disease.csv").set_index('id')
