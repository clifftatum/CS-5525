import pandas as pd
from matplotlib import pyplot as plt
from EDA import EDA
import seaborn as sb
import numpy as np
from scipy import optimize
np.random.seed(5525)

sb.set(color_codes=True)
pd.set_option("display.precision", 2)
pd.options.display.float_format = "{:,.2f}".format

if __name__ == '__main__':
    eda = EDA('Clifford Tatum Lecture 7- CS-5525')

    mean = 0
    var = 1
    N = 100
    x = np.linspace(-2,2,N)
    e = np.random.normal(mean, var, N)
    intercept = 2
    slope = 3

    y = intercept+slope*x+e

    fig = plt.figure
    plt.scatter(x,y,color = 'r')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    def objective(x,a,b):
        return a*x +b

    popt, _ = optimize.curve_fit(objective,x,y)
