import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')

data = pd.read_csv('nocd.csv')

data.drop(['STATION', 'NAME'], axis=1, inplace=True)
data.fillna(data.mean())

data_for_prediction = data.drop('DATE', axis=1)

print(data)
print(data_for_prediction)
