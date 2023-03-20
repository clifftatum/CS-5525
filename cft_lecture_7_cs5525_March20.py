import pandas as pd
from matplotlib import pyplot as plt
from src.EDA import EDA
import seaborn as sb
import numpy as np
import statsmodels.api as sm
np.random.seed(123)

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lecture 7 - March 20 CS-5525')
    X = np.array([[1,1],[1,2],[1,3],[1,4],[1,5]])
    n=5
    p=1
    Y = np.array([[2],[4],[5],[4],[5]])

    ###################################################################################################################
    # Ordinary Least Squares Regression
    ###################################################################################################################
    # Manual

    H = X.T @ X
    Beta = np.linalg.inv(H) @ X.T @ Y
    y_hat = X @ Beta
    e  = Y-y_hat
    SSE = e.T @ e
    var_e = SSE/(n-p-1)
    sigma_e = var_e**(0.5)


    X_star_1 = X[0,:]
    upper = Y[0] + 1.96*sigma_e*(1+X_star_1 @ (np.linalg.inv(H) @ X_star_1.T))**(0.5)
    lower = Y[0] - 1.96*sigma_e*(1+X_star_1 @ (np.linalg.inv(H) @ X_star_1.T))**(0.5)

    X_star_2 = X[1, :]
    upper2 = Y[1] + 1.96*sigma_e*(1+X_star_2 @ (np.linalg.inv(H) @ X_star_2.T))**(0.5)
    lower2 = Y[1] - 1.96*sigma_e*(1+X_star_2 @ (np.linalg.inv(H) @ X_star_2.T))**(0.5)

    X_star_3 = X[2, :]
    upper3 = Y[2] + 1.96*sigma_e*(1+X_star_3 @ (np.linalg.inv(H) @ X_star_3.T))**(0.5)
    lower3 = Y[2] - 1.96*sigma_e*(1+X_star_3 @ (np.linalg.inv(H) @ X_star_3.T))**(0.5)

    X_star_4 = X[2, :]
    upper4 = Y[3] + 1.96*sigma_e*(1+X_star_4 @ (np.linalg.inv(H) @ X_star_4.T))**(0.5)
    lower4 = Y[3] - 1.96*sigma_e*(1+X_star_4 @ (np.linalg.inv(H) @ X_star_4.T))**(0.5)

    X_star_5 = X[4, :]
    upper5 = Y[4] + 1.96*sigma_e*(1+X_star_5 @ (np.linalg.inv(H) @ X_star_5.T))**(0.5)
    lower5 = Y[4] - 1.96*sigma_e*(1+X_star_5 @ (np.linalg.inv(H) @ X_star_5.T))**(0.5)
    Y_Y_hat = pd.DataFrame(np.hstack((Y,y_hat)))
    yy_hat_corr = Y_Y_hat.corr()
    R_squared = yy_hat_corr.iloc[0,p]**2

    ####################################################################################################################
    # package
    X = [1,2,3,4,5]
    Y = [2,4,5,4,5]
    X = np.array(X)
    Y = np.array(Y)
    X = sm.add_constant(X)
    model = sm.OLS(Y,X).fit()
    print(model.summary())

    ####################################################################################################################
    # Singular Value Decomposition
    ####################################################################################################################

    x1 = np.random.normal(0,1,1000)
    x2 = np.random.normal(0, 1, 1000)
    x3 = np.random.normal(0, 1, 1000)
    x4 = 2*x1#np.random.normal(0, 1, 1000)
    X = np.vstack((x1,x2,x3,x4)).T # [nxp]
    print(X.shape)
    _,d,_ = np.linalg.svd(X)
    print(d) # small is co-linear
    # last should go to zeros, as x4 is co-linear with x1
    # SVD tells me there is a problem, but not one which one to drop!!!
    print(f'condition number is {np.linalg.cond(X)}') # large is co-linear

    ####################################################################################################################

    ####################################################################################################################






    pass
