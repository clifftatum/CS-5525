import warnings

import pandas as pd
from src.EDA import EDA
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# Ensure with latest tensorflow: pip install numpy==1.21

pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum exploratoryDataAnalysis Deflation Market Nov 2022 - Jan 2023 - CS-5525')
    url = 'https://raw.githubusercontent.com/clifftatum/CS-5525-Term-Project/main/DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv'
    df = pd.read_csv(url)


    # OUTPUT LAYER MUST BE SAME DIMENSION OF FEATURESET AND SIGMOID
    # BATCH_SIZE_ FRACTION OF TOTAL NUMBER OF OBSERVATIONS

    pass
