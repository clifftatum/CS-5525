import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
np.random.seed(5525)
N = 100
e = np.random.normal(0,1,N)
x = np.linspace(-2,2,N)
intercept, slope = 2,3
y1 = intercept + slope * x
y = y1 + e



def objective(x, a, b):
    return a*x + b

popt, _  = curve_fit(objective, x,y)
print(popt)
a = popt[0]
b = popt[1]

x_line = np.linspace(min(x), max(x), len(x))
y_line = objective(x_line, a, b)

plt.figure()
plt.scatter(x,y, color = 'red')
plt.plot(x_line, y_line, '--', color = 'blue', lw  =3)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.show()