import os
import librosa
import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc

# to convert between integers and strings
digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

digit_values = {'zero': 0, 'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9}

def generate_metadata(dirname='speech_commands_v0.01',
                      separator='/',
                      metadata_filename='metadata.csv'):
    """
    Function to generate metadata, if you haven't downloaded metadata.csv. Consists of:

    filename: the path from this notebook's directory to the audio file in question.
    rec_id: hash code for the filename, which serves as an int index.
    digit: the name of the digit being spoken (which is the name of the subdirectory
           containing the file)
    length: the length of the file, in number of samples at a sampling rate of 8000

    Saves all metadata in the file metadata_filename. If you've downloaded metadata.csv,
    there is no need to run this function.

    This function does not return anything.

    ==========================================================================================

    Parameters
    ----------
    dirname (str):
        The name of the directory containing the speech commands.
        Default: 'speech_commands_v0.01'

    separator (str):
        Delimiting character for the filepath.
        Default: '/'

    metadata_filename (str):
        The name of the file where metadata will be stored. If extension '.csv'
        is not specified, '.csv' will be appended.
        Default: 'metadata.csv'

    ==========================================================================================

    Returns
    -------
    None
    """

    # these are the digits we're working with, which also have to
    digits = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']

    # generate the name of each file
    filenames = [dirname
                 + separator
                 + digit
                 + separator
                 + filename
                 for digit in digits for filename in os.listdir(dirname + '/' + digit)]

    # check to make sure that the number of unique hash codes matches the number of unique filenames.
    # in other words, each recording can be identified uniquely by
    assert len(filenames) == len(set(hash(filename) for filename in filenames)), 'colliding hash codes'

    # generate the metadata for each of the files.
    # metadata includes:
    #   filename: the complete path to the file from the root directory
    #   rec_id: the hash code for that particular filename
    #   digit: the name of the digit spoken in the recording
    #   length: the length of the recording, when sampled at 8000 Hz
    metadata = [{'filename':filename,
                 'rec_id':hash(filename),
                 'digit':filename.split(separator)[1],
                 'length':len(librosa.load(filename, sr=8000)[0])} for filename in tqdm.tqdm(filenames)]

    # append extension if it is not already specified
    if not metadata_filename.endswith('.csv'):
        metadata_filename += '.csv'

    # create dictionary from the metadata
    # create DataFrame from metadata with columns in consistent order
    # save DataFrame as a csv with name metadata_filename
    pd.DataFrame.from_dict(metadata)[['filename', 'rec_id', 'digit', 'length']].to_csv(metadata_filename)

import pandas as pd
from sklearn.model_selection import train_test_split

def split_metadata(digits=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
                   filename='metadata.csv',
                   test_size=0.2,
                   random_state=0,
                   shuffle=True):
    """
    Most arguments are passed directly into train_test_split, and behave the same way.
    Splitting is performed on metadata.csv, which contains the following attributes:
        rec_id: a unique identifier for the recording
        filename: a complete filepath to the recording
        digit: a string containing the digit being spoken
        length: the length in terms of the number of samples (sample rate: 8000/sec)

    Only returns metadata for the classes specified in the digits parameter. Must contain
    at least one of the values from the default list, or a ValueError will be thrown. As long
    as there is at least one valid value, all invalid values will be ignored. Default
    loads all ten classes.

    Returns train_test_split's return values for only the digits specified by the digits
    argument.

    One-hots the response labels and returns Y as a sparse matrix.

    ==========================================================================================

    Parameters
    ----------
    digits (list-like):
        A list of digits to include in the train/test/split data. All valid digits are
        included in the default list.
        Default: ['zero', 'one', 'two', 'three', 'four',
                  'five', 'six', 'seven', 'eight', 'nine']

    filename (str):
        The name of the file where the metadata is stored.
        Default: 'metadata.csv'

    test_size (float):
        Specifies the proportion of the data to put in the 'test' list. Passed directly to
        train_test_split.
        Default: 0.2

    random_state (int):
        A seed for the random number generator. Passed directly to train_test_split.
        Default: 0

    shuffle (bool):
        Whether to shuffle the data before splitting into training and test sets.
        Default: True

    ==========================================================================================

    Returns
    -------
    X_train (pandas.core.series.Series):
        List of recordings for training data. Index is id, value is filepath. Represents
        100 * (1 - test_size)% of the data in the specified classes.

    X_test (pandas.core.series.Series):
        List of recordings for test data. Index is id, value is filepath. Represents
        100 * test_size% of the data in the specified classes.

    y_train (pandas.core.frame.DataFrame):
        One-hotted classes for the training data. Index is id, each column is a digit.
        Column order is arbitrary. Represents 100 * (1-test_size)% of the data.

    y_test (pandas.core.frame.DataFrame):
        One-hotted classes for the test data. Index is id, each column is a digit.
        Columnn order is arbitrary. Represents 100 * test_size% of the data.
    """

    # read metadata into a dataframe and index by the recording's id
    print(f'Reading metadata from {filename}')
    mdf = pd.read_csv('metadata.csv')[['filename', 'rec_id', 'digit', 'length']].set_index('rec_id')

    # only keep the recordings that are in the digits parameter
    # if no valid digits are specified in this parameter, this line will generate
    # a ValueError
    print('Subsetting to', digits)
    mdf = mdf[mdf.digit.isin(digits)]

    # Subset to only the recordings that are exactly 8000 samples long (1 second)
    # This is approximately 91% of the full dataset across ten classes, and each
    # class makes up between 9.7% and 10.3% of the recordings.
    print('Subsetting for uniform length...')
    mdf = mdf[mdf.length == 8000]

    # keep only the filename column for the training and test data
    print('Keeping only the filename...')
    X = mdf.filename

    # one-hot encode the classes for classifying
    print('One-hot encoding the target variable...')
    Y = pd.get_dummies(mdf.digit)[digit_names]

    # call sklearn.model_selection's train_test_split, pass it the arguments passed
    # to split_metadata and directly return the return values
    return train_test_split(X,
                            Y,
                            test_size=test_size,
                            random_state=random_state,
                            shuffle=shuffle)

def load_raw_data(digits=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
                  test_size=0.2,
                  random_state=0,
                  only_train=False,
                  shuffle=True):
    """
    Returns the raw data at a sampling rate of 8000. No transformation have been performed yet.

    Only loads data for the classes specified in the digits parameter. Must contain
    at least one of the values from the default list, or a ValueError will be thrown. As long
    as there is at least one valid value, all invalid values will be ignored. Default
    loads all ten classes.

    Loads each file using the filename in the training metadata and test metadata, then recompiles
    it into a DataFrame with the original index.

    This function also provides an option for loading only the training data (which will be
    cross-validated in each epoch of the neural network).

    ==========================================================================================

    Parameters
    ----------
    digits (list-like):
        A list of digits to include in the train/test/split data. All valid digits are
        included in the default list.
        Default: ['zero', 'one', 'two', 'three', 'four',
                  'five', 'six', 'seven', 'eight', 'nine']

    test_size (float):
        Specifies the proportion of the data to put in the 'test' list. Passed directly to
        train_test_split.
        Default: 0.2

    random_state (int):
        A seed for the random number generator. Passed directly to train_test_split.
        Default: 0

    shuffle (bool):
        Whether to shuffle the data before splitting into training and test sets.
        Default: True

    only_train (bool):
        If this is selected, only the training data is loaded. Accuracy is measured with
        cross-validation at the end of each epoch.

    ==========================================================================================

    Returns
    -------
    X_train (pandas.core.frame.DataFrame):
        DataFrame will vary in number of rows, depending on which digit classes are subset
        and what the training/test size is.

        Has 8000 columns (0-7999), representing the sequence of samples taken.

        Index is the unique id given to each recording (order should be preserved)

    X_test (pandas.core.frame.DataFrame):
        DataFrame will vary in number of rows, depending on which digit classes are subset
        and what the training/test size is.

        Has 8000 columns (0-7999), representing the sequence of samples taken.

        Index is the unique id given to each recording (order should be preserved)

    y_train (pandas.core.frame.DataFrame):
        One-hotted classes for the training data. Index is id, each column is a digit.
        Column order is arbitrary. Represents 100 * (1-test_size)% of the data.

    y_test (pandas.core.frame.DataFrame):
        One-hotted classes for the test data. Index is id, each column is a digit.
        Columnn order is arbitrary. Represents 100 * test_size% of the data.
    """
    print('Loading raw data...')

    X_train = dict()
    X_test = dict()

    print('Loading metadata...')
    X_train_md, X_test_md, y_train, y_test = split_metadata(digits=digits,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            shuffle=shuffle)

    # regardless of the value of only_train, we need to load the training data
    print('Loading raw training data from wav files...')
    for uid_train in tqdm.tqdm(X_train_md.index, desc='loading training data'):
        train_obs, sr = librosa.load(X_train_md.loc[uid_train], sr=8000)
        X_train[uid_train] = train_obs

    print('Constructing training DataFrame... this might take a few minutes')
    X_train = pd.DataFrame.from_dict(X_train, orient='index')

    # if only_train is True, we can speed things up by not loading the test data
    if only_train == True:
        return X_train, y_train

    else: # otherwise, we need to load the test data
        print('Loading raw test data from wav files...')
        for uid_test in tqdm.tqdm(X_test_md.index, desc='    loading test data'):
            test_obs, sr = librosa.load(X_test_md.loc[uid_test], sr=8000)
            X_test[uid_test] = test_obs

        print('Constructing test DataFrame...')
        X_test = pd.DataFrame.from_dict(X_test, orient='index')
        return X_train, X_test, y_train, y_test

def load_stft_data(digits=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
                   test_size=0.2,
                   random_state=0,
                   only_train=False,
                   shuffle=True,
                   mean_of='freq'):
    """
    Samples data at 8000 Hz, then performs the following transformations:

    1. Computes the stft, converting the raw 8000x1 array into a 1025x16 array
    2. Takes the absolute value of each entry to convert complex numbers to real numbers.
    3. Takes the average of each frequency bin to remove time and reduce dimensionality.
       This converts each observation into a 1025x1 array.

    Only loads data for the classes specified in the digits parameter. Must contain
    at least one of the values from the default list, or a ValueError will be thrown. As long
    as there is at least one valid value, all invalid values will be ignored. Default
    loads all ten classes.

    Loads each file using the filename in the training metadata and test metadata, then recompiles
    it into a DataFrame with the original index.

    This function also provides an option for loading only the training data (which will be
    cross-validated in each epoch of the neural network).

    ==========================================================================================

    Parameters
    ----------
    digits (list-like):
        A list of digits to include in the train/test/split data. All valid digits are
        included in the default list.
        Default: ['zero', 'one', 'two', 'three', 'four',
                  'five', 'six', 'seven', 'eight', 'nine']

    test_size (float):
        Specifies the proportion of the data to put in the 'test' list. Passed directly to
        train_test_split.
        Default: 0.2

    random_state (int):
        A seed for the random number generator. Passed directly to train_test_split.
        Default: 0

    only_train (bool):
        If this is selected, only the training data is loaded. Accuracy is measured with
        cross-validation at the end of each epoch.
        Default: False

    shuffle (bool):
        Whether to shuffle the data before splitting into training and test sets.
        Default: True

    mean_of (str):
        Whether to take the mean of each frequency bin or of each time window.
        Acceptable values are 'freq' and 'time.' (Defaults to 'freq' if the input
        is not understood.)
        Default: 'freq'

    ==========================================================================================

    Returns
    -------
    X_train (pandas.core.frame.DataFrame):
        DataFrame will vary in number of rows, depending on which digit classes are subset
        and what the training/test size is.

        Has 8000 columns (0-7999), representing the sequence of samples taken.

        Index is the unique id given to each recording (order should be preserved)

    X_test (pandas.core.frame.DataFrame):
        DataFrame will vary in number of rows, depending on which digit classes are subset
        and what the training/test size is.

        Has 8000 columns (0-7999), representing the sequence of samples taken.

        Index is the unique id given to each recording (order should be preserved)

    y_train (pandas.core.frame.DataFrame):
        One-hotted classes for the training data. Index is id, each column is a digit.
        Column order is arbitrary. Represents 100 * (1-test_size)% of the data.

    y_test (pandas.core.frame.DataFrame):
        One-hotted classes for the test data. Index is id, each column is a digit.
        Columnn order is arbitrary. Represents 100 * test_size% of the data.
    """
    print('Loading stft data...')

    X_train = dict()
    X_test = dict()

    # a variable that determines which axis to take the
    mean_axis = 0 if mean_of == 'time' else 1

    print('Loading metadata...')
    X_train_md, X_test_md, y_train, y_test = split_metadata(digits=digits,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            shuffle=shuffle)

    # regardless of the value of only_train, we need to load the training data
    print('Computing mean of stfts by {}...'.format(mean_of))
    for uid_train in tqdm.tqdm(X_train_md.index, desc='loading training data'):
        train_obs, sr = librosa.load(X_train_md.loc[uid_train], sr=8000)
        X_train[uid_train] = abs(librosa.stft(train_obs)).mean(axis=mean_axis).T

    print('Constructing training DataFrame... this might take a few minutes')
    X_train = pd.DataFrame.from_dict(X_train, orient='index')

    # if only_train is True, we can speed things up by not loading the test data
    if only_train == True:
        return X_train, y_train

    else: # otherwise, we need to load the test data
        print('Loading raw test data from wav files...')
        for uid_test in tqdm.tqdm(X_test_md.index, desc='    loading test data'):
            test_obs, sr = librosa.load(X_test_md.loc[uid_test], sr=8000)
            X_test[uid_train] = abs(librosa.stft(test_obs)).mean(axis=mean_axis).T

        print('Constructing test DataFrame...')
        X_test = pd.DataFrame.from_dict(X_test, orient='index')
        return X_train, X_test, y_train, y_test

def load_mfcc_data(digits=['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'],
                   test_size=0.2,
                   random_state=0,
                   only_train=False,
                   shuffle=True):
    """
    Samples data at 8000 Hz, then uses librosa's mfcc function to extract the
    Mel-frequency ceptstral coefficients (MFCCs).

    Only loads data for the classes specified in the digits parameter. Must contain
    at least one of the values from the default list, or a ValueError will be thrown. As long
    as there is at least one valid value, all invalid values will be ignored. Default
    loads all ten classes.

    Loads each file using the filename in the training metadata and test metadata, then recompiles
    it into a DataFrame with the original index.

    This function also provides an option for loading only the training data (which will be
    cross-validated in each epoch of the neural network).

    ==========================================================================================

    Parameters
    ----------
    digits (list-like):
        A list of digits to include in the train/test/split data. All valid digits are
        included in the default list.
        Default: ['zero', 'one', 'two', 'three', 'four',
                  'five', 'six', 'seven', 'eight', 'nine']

    test_size (float):
        Specifies the proportion of the data to put in the 'test' list. Passed directly to
        train_test_split.
        Default: 0.2

    random_state (int):
        A seed for the random number generator. Passed directly to train_test_split.
        Default: 0

    only_train (bool):
        If this is selected, only the training data is loaded. Accuracy is measured with
        cross-validation at the end of each epoch.
        Default: False

    shuffle (bool):
        Whether to shuffle the data before splitting into training and test sets.
        Default: True

    ==========================================================================================

    Returns
    -------
    X_train (pandas.core.frame.DataFrame):
        DataFrame will vary in number of rows, depending on which digit classes are subset
        and what the training/test size is.

        Has 8000 columns (0-7999), representing the sequence of samples taken.

        Index is the unique id given to each recording (order should be preserved)

    X_test (pandas.core.frame.DataFrame):
        DataFrame will vary in number of rows, depending on which digit classes are subset
        and what the training/test size is.

        Has 8000 columns (0-7999), representing the sequence of samples taken.

        Index is the unique id given to each recording (order should be preserved)

    y_train (pandas.core.frame.DataFrame):
        One-hotted classes for the training data. Index is id, each column is a digit.
        Column order is arbitrary. Represents 100 * (1-test_size)% of the data.

    y_test (pandas.core.frame.DataFrame):
        One-hotted classes for the test data. Index is id, each column is a digit.
        Columnn order is arbitrary. Represents 100 * test_size% of the data.
    """
    print('Loading MFCC data...')

    X_train = []
    X_test = []

    print('Loading metadata...')
    X_train_md, X_test_md, y_train, y_test = split_metadata(digits=digits,
                                                            test_size=test_size,
                                                            random_state=random_state,
                                                            shuffle=shuffle)

    # regardless of the value of only_train, we need to load the training data
    print('Computing mfccs...')
    for uid_train in tqdm.tqdm(X_train_md.index, desc='loading training data'):
        train_obs, sr = librosa.load(X_train_md.loc[uid_train], sr=8000)
        #X_train.append(librosa.feature.mfcc(train_obs))
        X_train.append(mfcc(train_obs))

    print('Converting training data to numpy array...')
    X_train = np.array(X_train)

    # if only_train is True, we can speed things up by not loading the test data
    if only_train == True:
        return X_train, y_train

    else: # otherwise, we need to load the test data
        print('Loading raw test data from wav files...')
        for uid_test in tqdm.tqdm(X_test_md.index, desc='    loading test data'):
            test_obs, sr = librosa.load(X_test_md.loc[uid_test], sr=8000)
            X_test.append(librosa.feature.mfcc(test_obs))
            X_test.append(mfcc(test_obs))

        print('Converting test data to numpy array...')
        X_test = np.array(X_test)
        return X_train, X_test, y_train, y_test
