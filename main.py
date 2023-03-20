import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import numpy as np

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum EDA Deflation Market Nov 2022 - Jan 2023 - CS-5525')
    url = 'https://raw.githubusercontent.com/clifftatum/CS-5525-Term-Project/main/DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv'
    df = pd.read_csv(url)

    # prune the perc increase feature as filling via mean or interpolation wouldn't make sense
    df.drop('')
    pass
