from src.function import *
import numpy as np


if __name__ == '__main__':
    a = np.eye(4)
    b = np.eye(4)
    b[2] = np.array([0, 0, 1, 1])
    b[3] = np.array([0, 0, 1, 0])

    for i in range(4):
        print roc_auc_multi(a, b, [i], np.mean)
        print roc_auc_multi(a, b, [i], np.median)
        print enrichment_factor_multi(a, b, 0.5)
        print