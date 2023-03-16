import pandas as pd
from matplotlib import pyplot as plt
import scipy.stats as st
pd.set_option("display.precision", 2)
from EDA import EDA
import seaborn as sb
import cv2
sb.set(color_codes=True)
import numpy as np
np.random.seed(5525)
from sklearn.datasets import make_regression,make_classification,make_blobs
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.ensemble import RandomForestRegressor
if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lecture 5 - Feb/13/2023 - CS-5525')
    url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Train_UWu5bXk.csv'
    plot_show = True
    df = pd.read_csv(url)
    #############################
    # STANDARDIZATION
    #############################
    mean = 194
    std = 11.2
    critical_val1 = 225
    critical_val2 = 175

    print(f'The probability that the observation'
          f' is between the 2 critical values ({critical_val1} , {critical_val2}) is '
          f'{(st.norm(mean,std).cdf(critical_val1) -st.norm(mean,std).cdf(critical_val2))*100:.2f}%')
    bot = st.norm.cdf(-1.69)
    top = st.norm.cdf(2.76)


    x = np.array([13,16,19,22,23,38,47,56,58,63,65,71])
    z = (x - np.mean(x))/np.std(x)
    print(z)
    scalar = StandardScaler() # creates an env for standardization
    scalar.fit(x.reshape(-1,1)) # how many rows do I have, -1 will decide (I know n columns)
    scalar_transform = scalar.transform(x.reshape(-1,1))
    print(f'standardized data:\n {np.round(scalar_transform)}')
    # note how z = scalar_transform (manual vs package)

    #############################
    # NORMALIZATION
    #############################

    normalized = (x - np.min(x)) / (np.max(x) - np.min(x))

    median_scale_by_quantile = (x-np.median(x))/(np.quantile(x,.75) - np.quantile(x,.25))
    if plot_show:
        plt.plot(normalized, label="Normalized")
        plt.plot(z, label="Standardized")
        plt.plot(median_scale_by_quantile,label = 'IQ Transformation')
        plt.xlabel('Number of Observations')
        plt.ylabel('Magnitude')
        plt.title('Standardization vs. Normalization')
        plt.legend()
        plt.show()




    pass





