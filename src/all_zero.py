import numpy as np

def all_zero_short_circuit(y_train):
    y = np.asarray(y_train, float).ravel()
    return np.all(y == 0)
