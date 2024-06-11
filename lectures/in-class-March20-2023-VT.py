import statsmodels.api as sm
import numpy as np
np.random.seed(123)

x1 = np.random.normal(0,1,1000)
x2 = np.random.normal(0,1,1000)
x3 = np.random.normal(0,1,1000)
x4 = 2*x1#np.random.normal(0,1,1000)

X = np.vstack((x1,x2,x3,x4)).T

_,d,_ = np.linalg.svd(X)
print(d)
print(f'condition number is {np.linalg.cond(X):.2f}')

X = [1,2,3,4,5]
Y = [2,4,5,4,5]

X = np.array(X)
Y = np.array(Y)

X = sm.add_constant(X)
# print(X)
model = sm.OLS(Y,X).fit()
print(model.summary())