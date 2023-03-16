import pandas as pd
from matplotlib import pyplot as plt
from EDA import EDA
import seaborn as sb
import plotly.express as px
import numpy as np

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum EDA Deflation Market Nov 2022 - Jan 2023 - CS-5525')
    url = 'https://raw.githubusercontent.com/clifftatum/CS-5525-Term-Project/main/DELIVER_CS5525_term_dataset_cliffordt_Deflation_EcoEnv_Nov2022_Jan2023.csv'
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True,ascending = True)

    eda.get_percent_of_missing_observations(df,clean=False,show=True,clean_method=None)


    fig = px.scatter(df, x="SlowPctK", y="PctB", color="target_buy_sell_performance", marginal_y="violin",
                     marginal_x="box", trendline="ols", template="simple_white")
    fig.show()

    df.drop(['price_pct_change'],axis1,inplace=True)

    pass
