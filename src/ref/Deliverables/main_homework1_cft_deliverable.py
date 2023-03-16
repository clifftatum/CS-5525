import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import numpy as np
from sklearn.datasets import make_regression,make_classification,make_blobs


sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':

    show_plot = True
    save_plot = False

    # Problem 1
    x_reg, y_reg = make_regression(n_samples=1000,
                                   n_features=100,
                                   n_informative=100,
                                   n_targets=1,
                                   random_state=5525)# must be fixed to 5525
    # Make a DataFrame, display first rows
    x_temp_df = pd.DataFrame(x_reg)
    y_temp_df = pd.DataFrame(y_reg)
    print('feature matrix: ')
    print(x_temp_df.head())

    print('target matrix: ')
    print(y_temp_df.head())

    # Problem 2
    x_temp_df.columns=[f'feature {i}' for i in range(1, x_reg.shape[1] + 1)]
    y_temp_df.columns=['target']

    df_reg = pd.concat([x_temp_df,y_temp_df],axis = 1)
    print('last 5 rows of concatenated feature and target regression dataframe: ')
    print(df_reg.tail())

    # Problem 3
    print('first 5 rows of sliced dataframe:')
    df_reg_sliced = df_reg.iloc[:,:5]
    print(df_reg.iloc[:5,:5].to_string())

    # Problem 4
    print('Covariance matrix of 5x5 sliced dataframe')
    print(df_reg_sliced.cov())
    print('Correlation matrix of 5x5 sliced dataframe')
    print(df_reg_sliced.corr())

    # Problem 5
    sb.set_style("ticks")
    pbd_ax = sb.pairplot(df_reg_sliced,kind='kde')

    # if save_plot:
    #     plt.savefig('P5.png')


    # Problem 6
    fig = plt.figure()
    plt.scatter(df_reg['feature 1'],df_reg['target'],label='feature 1 vs. target')
    plt.xlabel('feature 1', weight='bold', size=12)
    plt.ylabel('target', weight='bold', size=12)
    plt.title('Regression synthetic dataset', weight='bold', size=12)
    plt.legend()
    plt.grid(color='k', linestyle='-')
    # if save_plot:
    #     plt.savefig('P6.png')

    # Problem 7
    x_class,y_class = make_classification(n_samples=1000,
                            n_features=100,
                            n_informative=100,
                            n_redundant=0,
                            n_repeated =0,
                            n_classes=4,
                            random_state=5525) # must be fixed to 5525

    # Construct a dataFrame, display first rows
    concat_class = np.hstack((x_class,y_class.reshape(len(y_class),1)))
    df_class = pd.DataFrame(concat_class)
    df_class.columns=[f'feature {i}' if i <= df_class.shape[1]-1 else 'target' for i in range(1, df_class.shape[1] + 1)]
    print('first 5 rows of concatenated feature and target classification dataframe: ')
    print(df_class.head())
    print('last 5 rows of concatenated feature and target classification dataframe: ')
    print(df_class.tail())

    # Problem 8
    sb.set_style("ticks")
    pbd_ax = sb.pairplot(df_class.iloc[:,:5],kind='kde')
    # pbd_ax.map_lower(sb.kdeplot, levels=4, color=".2")
    # pbd_ax.map_upper(sb.kdeplot, levels=4, color=".2")
    # if save_plot:
    #     plt.savefig('P8.png')


    # Problem 9
    x_blob,y_blob = make_blobs(n_samples=5000,
                            centers=4,
                            n_features=2,
                            random_state=5525)# must be fixed to 5525

    concat_blob = np.hstack((x_blob,y_blob.reshape(len(y_blob),1)))
    df_blob = pd.DataFrame(concat_blob)
    df_blob.columns=[f'feature {i}' if i <= df_blob.shape[1]-1 else 'target' for i in range(1, df_blob.shape[1] + 1)]

    # Problem 10
    plt.figure()
    sb.scatterplot(data=df_blob, x='feature 1', y='feature 2', hue='target',palette = "husl")
    plt.xlabel('feature 1', weight='bold', size=12)
    plt.ylabel('feature 2', weight='bold', size=12)
    plt.title('The isotropic Gaussian blob with 4 centers and 2 features', weight='bold', size=12)
    plt.legend()
    plt.grid(color='k', linestyle='-')
    if show_plot:
        # if save_plot:
        #     plt.savefig('P10.png')
        plt.show()


