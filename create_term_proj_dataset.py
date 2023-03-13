import pandas as pd
from matplotlib import pyplot as plt
from EDA import EDA
import seaborn as sb
import numpy as np


sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

#######################################################################################################################
# This scrips takes a CSV of N number of various publicly trade-able symbol observations along with various stochastic
# features and evalutes the performance of the BUY or SELL signal against a single stochastic cycle's worth of fiscal
# performance - this metric is then categorized to a target feature in another script
#######################################################################################################################






if __name__ == '__main__':
    eda = EDA('Clifford Tatum Term Project - DATASET CREATION - CS-5525')
    url = 'https://raw.githubusercontent.com/clifftatum/CS-5525-Term-Project/main/Nov22_Jan31_Daily.csv'
    df = pd.read_csv(url, encoding= 'unicode_escape')

    # Prune the features not included in our discussion
    df_new = eda.slice_by_observation(df,
                                      feature=['PriceDate','Symbol','StockType','Sector','Industry',
                                               'SlowPctK','SlowPctD','PctB','SSCrossover',
                                               'SSFailure','SSEmbedded','MFI','RSI',
                                                'Open','Price','Low','High','LowerBollinger',
                                               'UpperBollinger','MovingAverage','Volume',
                                               'BandWidth','VolumeTenMA','IsFavorite','IsActive'],
                                      observations=None,
                                      obs_by_feature= None)



    # There are NaNs being reported as - , replace them with I for Intermediate
    df_new.replace('\-', 'I', regex=True,inplace=True)

    # Take a look at the SSCrossover metric
    df_look_1 = eda.slice_by_observation(df_new,
                                         feature=['PriceDate', 'Symbol','StockType', 'Sector', 'Industry',
                                                  'SlowPctK', 'SlowPctD', 'PctB', 'SSCrossover',
                                                  'SSFailure', 'SSEmbedded', 'MFI', 'RSI',
                                                  'Open', 'Price', 'Low', 'High', 'LowerBollinger',
                                                  'UpperBollinger', 'MovingAverage', 'Volume',
                                                  'BandWidth', 'VolumeTenMA', 'IsFavorite', 'IsActive'],
                                          observations=['A'],
                                          obs_by_feature= ['Symbol'])



    # Make sure your date column is a date an not a string
    df_new['DateTime'] = pd.to_datetime(df_new['PriceDate'])
    # df_new = df_new[(df_new['DateTime'] < '2023-3-1')]

    df_look_1 = df_new

    # The target  will be based on the stock pct increase/ decrease over one stochastic cycle
    def Stochastic(data, k_period: int = 14, d_period: int = 3, smooth_k=3,
                   names: tuple = ('OPEN', 'CLOSE', 'LOW', 'HIGH'), return_df: bool = False):
        '''
        Implementation of the Stochastic Oscillator. Returns the Fast and Slow lines values or the whole DataFrame
        args:
            data: Pandas Dataframe of the stock
            k_period: Period for the %K /Fast / Blue line
            d_period: Period for the %D / Slow /Red / Signal Line
            smooth_k: Smoothening the Fast line value. With increase/ decrease in number, it becomes the Fast or Slow Stochastic
            names: Names of the columns which contains the corresponding values
            return_df: Whether to return the DataFrame or the Values
        out:
            Returns either the Array containing (fast_line,slow_line) values or the entire DataFrame
        '''
        OPEN, CLOSE, LOW, HIGH = names
        df = data.copy()
        if df.iloc[0, 0] > df.iloc[
            1, 0]:  # if the first Date entry [0,0] is > previous data entry [1,0] then it is in descending order, then reverse it for calculation
            df.sort_index(ascending=False, inplace=True)

        # Adds a "n_high" column with max value of previous 14 periods
        df['n_high'] = df[HIGH].rolling(k_period).max()

        # Adds an "n_low" column with min value of previous 14 periods
        df['n_low'] = df[LOW].rolling(k_period).min()

        # Uses the min/max values to calculate the %k (as a percentage)
        df['SlowPctK'] = (df[CLOSE] - df['n_low']) * 100 / (df['n_high'] - df['n_low'])  # %K or so called Fast Line

        if smooth_k > 1:  # Smoothen the fast, blue line
            df['SlowPctK'] = df['SlowPctK'].rolling(smooth_k).mean()

        # Uses the %k to calculates a SMA over the past 3 values of %k
        df['SlowPctD'] = df['SlowPctK'].rolling(d_period).mean()  # %D of so called Slow Line

        df.drop(['n_high', 'n_low'], inplace=True, axis=1)

        df.sort_index(ascending=True, inplace=True)

        if return_df:
            return df

        return df.iloc[0, -2:]  # Fast


    def bollinger_band_upper(series: pd.Series, length: int = 20, *, num_std: int = 2) -> pd.Series:
        rolling = series.rolling(length)
        if num_std == 0:
            bband = rolling.mean()  # Skips calculating std.
        else:
            bband = rolling.mean() + rolling.std(ddof=0) * num_std
        return bband

    def bollinger_band_lower(series: pd.Series, length: int = 20, *, num_std: int = 2) -> pd.Series:
        rolling = series.rolling(length)
        if num_std == 0:
            bband = rolling.mean()  # Skips calculating std.
        else:
            bband = rolling.mean() - rolling.std(ddof=0) * num_std
        return bband







    df_look_1['price_pct_change'] = None

    # create the price_change_percent_per_ST_cycle
    # loop through the DF, if SS crossover is I - Intermediate - the target will be Hold
    for index, row in df_look_1.iterrows():
        print(f'Processing: index : {index} , Symbol: {row["Symbol"]}')
        found = False
        if row['SSCrossover'] == 'Y': # buy signal
            for jj in np.arange(index,len(df_look_1)):
                if df_look_1['Symbol'][jj] != row['Symbol']:
                    found = False
                    break
                if df_look_1['SSCrossover'][jj] == 'S': # find next sell signal
                    # calculate perc change
                    found = True
                    temp = ((df_look_1['High'][jj]-row['High']) / row['High']) * 100
                    df_look_1['price_pct_change'][index] = temp
                    break
            if df_look_1['price_pct_change'][index] is None: # the dataset doesnt contain the entire stochastic cycle
                # make an online query and do the calc here
                start_date_temp = row['PriceDate'].split('/')  # "2022-09-25"
                start_date = start_date_temp[2] + '-' + start_date_temp[0] + '-' + start_date_temp[1]
                end_date = str(int(start_date_temp[2])) + '-' + str(int(start_date_temp[0])+1) + '-' + str(start_date_temp[1])
                # if int(start_date_temp[2])==2023 and int(start_date_temp[0]) ==2:
                end_date = '2023-03-10'

                stock_ticker = row['Symbol']
                start = pd.to_datetime([start_date]).astype(int)[0] // 10 ** 9  # convert to unix timestamp.
                end = pd.to_datetime([end_date]).astype(int)[0] // 10 ** 9  # convert to unix timestamp.
                url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock_ticker + '?period1=' + str(
                    start) + '&period2=' + str(end) + '&interval=1d&events=history'
                try:
                    df_symb = pd.read_csv(url)
                except:
                    continue
                df_stoch1 = Stochastic(df_symb, k_period=14, d_period=3, smooth_k=3,
                                       names=('Open', 'Close', 'Low', 'High'), return_df=True)

                # df_stoch1['PctB'] = ((df_stoch1['Close'] - bollinger_band_lower(series=df_stoch1['Close'])) \
                #                     / (bollinger_band_upper(series=df_stoch1['Close']- bollinger_band_lower(series=df_stoch1['Close'])))) * 100
                for zz in np.arange(0,len(df_stoch1)-1, 1):
                    measurement = 0
                    if (df_stoch1['SlowPctK'][zz] >=80) and(df_stoch1['SlowPctD'][zz] >= 80):
                        measurement = df_stoch1['Close'][zz]
                        break
                if measurement ==0:
                    zz = np.argmax(df_stoch1['Close'])
                    measurement = df_stoch1['Close'][zz]
                df_look_1['price_pct_change'][index] = ((measurement-row['High'])/row['High'])*100


        if row['SSCrossover'] == 'S':  # sell signal
            for ii in np.arange(index,0,-1):
                if df_look_1['Symbol'][ii] != row['Symbol']:
                    found = False
                    break
                if df_look_1['SSCrossover'][ii] == 'Y': # find previous Buy
                    # calculate perc change
                    found = True
                    temp = ((row['High'] - df_look_1['High'][ii]) / df_look_1['High'][ii]) * 100
                    df_look_1['price_pct_change'][index] = temp
                    break

            if df_look_1['price_pct_change'][index] is None: # the dataset doesnt contain the entire stochastic cycle
                # make an online query and do the calc here
                end_date_temp = row['PriceDate'].split('/') # "2022-09-25"
                end_date = end_date_temp[2]+'-'+end_date_temp[0]+'-'+end_date_temp[1]
                start_date = str(int(end_date_temp[2])-1) + '-'+str(end_date_temp[0] ) + '-'+str(end_date_temp[1])
                # start_date = "2022-10-01"

                stock_ticker = row['Symbol']
                start = pd.to_datetime([start_date]).astype(int)[0] // 10 ** 9  # convert to unix timestamp.
                end = pd.to_datetime([end_date]).astype(int)[0] // 10 ** 9  # convert to unix timestamp.
                url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock_ticker + '?period1=' + str(
                    start) + '&period2=' + str(end) + '&interval=1d&events=history'
                try:
                    df_symb = pd.read_csv(url)
                except:
                    continue
                df_stoch1 = Stochastic(df_symb,k_period=14,d_period=3,smooth_k=3,names = ('Open', 'Close', 'Low', 'High'),return_df=True)

                # df_stoch1['PctB'] = ((df_stoch1['Close'] - bollinger_band_lower(series=df_stoch1['Close'])) \
                #                     /( bollinger_band_upper(series=df_stoch1['Close'])- bollinger_band_lower(series=df_stoch1['Close'])))*100
                for kk in np.arange(len(df_stoch1)-1, 0,-1):
                    measurement =0
                    if ((df_stoch1['SlowPctK'][kk] >=20) and (df_stoch1['SlowPctK'][kk] <=25) and \
                       (df_stoch1['SlowPctD'][kk] >= 20) and (df_stoch1['SlowPctD'][kk] <= 25)) or \
                            ((df_stoch1['SlowPctK'][kk] <=23) and (df_stoch1['SlowPctD'][kk] <=23)):
                        measurement = df_stoch1['Close'][kk]
                        break
                df_look_1['price_pct_change'][index] = ((row['High'] - measurement)/measurement)*100

    # df_look_1.to_csv('deflation_market-dataset_2023_cliffordt_CS5525_term_dataset.csv')
