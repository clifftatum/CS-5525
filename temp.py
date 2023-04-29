import statsmodels.api as sm
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
y = np.array(['A', 'B', 'C', 'A'])

# Convert to pandas dataframe
df = pd.DataFrame(X, columns=['feature1', 'feature2', 'feature3'])
df['target'] = y

model = sm.OLS(y, X).fit(method='ordinal')

# Print the summary of the model
print(model.summary())