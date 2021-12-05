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

    # get labels
    y = ba['class']
    ba.drop(columns='class', inplace=True)

    # normalize data
    ba = (ba - ba.mean())/ba.std()

    return ba, y
