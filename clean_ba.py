def clean_ba(ba):
    '''
    Cleaning the Banknote Authentification dataset

    INPUT

    OUTPUT

    AUTHOR
    
    '''

    # get labels
    y = kd['class']
    kd.drop(columns='class', inplace=True)

    # normalize data
    ba = (ba - ba.mean())/ba.std()

    return ba, y
