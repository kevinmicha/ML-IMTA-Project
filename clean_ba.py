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
    ba.columns =['V1', 'V2', 'V3', 'V4', 'class']

    # Get labels
    y = ba['class']
    ba.drop(columns='class', inplace=True)

    # Normalize data
    ba = (ba - ba.mean())/ba.std()

    return ba, y
