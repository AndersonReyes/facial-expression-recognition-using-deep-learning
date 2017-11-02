import numpy as np
import csv
import os


def preprocess(X):
    # X_centered = X = np.mean(X, axis=1)
    X_normalized = np.divide(X, 255)
    return X_normalized


def load_data(path, expect_labels=True):
    assert path.endswith('.csv')
    # If a previous call to this method has already converted
    # the data to numpy format, load the numpy directly
    X_path = path[:-4] + '.X.npy'
    Y_path = path[:-4] + '.Y.npy'
    if os.path.exists(X_path):
        X = np.load(X_path)
        if expect_labels:
            y = np.load(Y_path)
        else:
            y = None
        return preprocess(X), y

    # Convert the .csv file to numpy
    csv_file = open(path, 'r')

    reader = csv.reader(csv_file)

    # Discard header
    row = next(reader)

    y_list = []
    X_list = []

    for row in reader:
        if expect_labels:
            y_str, X_row_str = (row[0], row[1])
            y = int(y_str)
            assert 0 <= y <= 6
            y_list.append(np.eye(7)[y])
        else:
            X_row_str = row[1]
        X_row_strs = X_row_str.split(' ')
        X_row = [float(x) for x in X_row_strs]
        X_list.append(X_row)

    X = np.asarray(X_list).astype('float32')
    if expect_labels:
        y = np.asarray(y_list)
    else:
        y = None

    np.save(X_path, X)
    if y is not None:
        np.save(Y_path, y)

    return preprocess(X), y
