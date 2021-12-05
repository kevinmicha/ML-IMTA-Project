# ========================================================================
# This file contains functions that clean and normalize two particular 
# datasets for a classification project at IMT Atlantique. Authors:  
# Martina BALBI, Mateo BENTURA, Ezequiel CENTOFANTI and Kevin MICHALEWICZ.
# ========================================================================

def clean_kd(kd):
    '''
    Cleaning the Kidney Disease dataset

    INPUT
    kd: The kidney disease dataframe

    OUTPUT
    kd: A cleaned version of the kidney disease dataframe
    y: A vector containing the classes

    AUTHOR
    Kevin Michalewicz
    '''
    # kd = pd.read_csv("kidney_disease.csv").set_index('id')

    kd[['pcv', 'rc', 'wc']] = kd[['pcv', 'rc', 'wc']].apply(pd.to_numeric, errors='coerce')

    kd.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                'anemia', 'classification']

    # Remove NaN
    numerical_columns = kd.columns[kd.dtypes=='float64']
    categorical_columns = kd.columns[kd.dtypes=='object']

    kd[numerical_columns] = kd[numerical_columns].fillna(kd[numerical_columns].mean())
    kd[categorical_columns] = kd[categorical_columns].fillna(kd[categorical_columns].mode().iloc[0])
    kd[categorical_columns] = kd[categorical_columns].replace(to_replace={'\t': '', ' ': ''}, regex=True)

    # Normalizing the data
    kd[numerical_columns] = (kd[numerical_columns] - kd[numerical_columns].mean()) / (kd[numerical_columns].std())

    # Cast labels to True or False
    kd = pd.get_dummies(kd, drop_first=True)

    # Extracting classes
    y = kd["classification_notckd"]
    kd = kd.drop(columns="classification_notckd")
    y = np.logical_xor(y,1).astype(int)

    return kd, y

def clean_ba(ba):
    '''
    Cleaning the Banknote Authentification dataset

    INPUT
    ba: the banknote authentication dataset

    OUTPUT
    ba: clean banknote authentication dataset
    y: labels

    AUTHOR
    Martina Balbi
    '''

    # Adding column names
    ba.columns = ['variance', 'skewness', 'curtosis', 'entropy', 'class']

    # Get labels
    y = ba['class']
    ba.drop(columns='class', inplace=True)

    # Normalize data
    ba = (ba - ba.mean())/ba.std()

    return ba, y
