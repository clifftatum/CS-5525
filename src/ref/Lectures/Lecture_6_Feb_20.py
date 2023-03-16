import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
pd.set_option("display.precision", 2)
from EDA import EDA
import seaborn as sb
sb.set(color_codes=True)
import numpy as np
np.random.seed(5525)
from sklearn.datasets import make_regression,make_classification,make_blobs
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.ensemble import RandomForestRegressor
if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lecture 6 - Feb/20/2023 - CS-5525')
    N = 1000
    dat = np.random.normal(0,1,N)

    m2 = eda.compute_skewness(dat,order=2)
    m3 = eda.compute_skewness(dat,order=3)
    g1 = m3/m2**(3/2)
    G1 = np.sqrt(N*(N-1)/(N-2))
    print(f'the skewness of the data is {g1:.2f}')
    print(f'the adjusted skewness of the data is {G1:.2f}')



