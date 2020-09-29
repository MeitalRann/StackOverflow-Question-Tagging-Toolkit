from sklearn.preprocessing import normalize
from scipy import stats
import numpy as np

a = np.array([[ 0.7972,  0.0767], [ 0.4383,  0.7866],  [0.8091,
               0.1954],  [0.6307,  0.6599],  [0.1065,  0.0508]])

mean = np.mean(a, axis=0)
std = np.std(a, axis=0)

b = stats.zscore(a, axis=0)
mean = np.mean(b, axis=0)
std = np.std(b, axis=0)
print('complete')
