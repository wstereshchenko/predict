import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


plt.style.use('ggplot')


def delete_passes(data_frame, name):

    for i in range(len(data_frame)):
        if np.isnan(data_frame[name][i]):

            p = i
            previous = False

            n = i
            following = False

            while not previous:
                p -= 1
                if not np.isnan(data_frame[name][p]):
                    previous = True

                if p == 0 and previous is False:
                    previous = True

            while not following:
                n += 1
                if not np.isnan(data_frame[name][n]):
                    following = True

                if n == len(data_frame)-1 and following is False:
                    following = True

            if np.isnan(data_frame[name][p]):
                data_frame[name][p] = data_frame[name][n]

            if np.isnan(data_frame[name][n]):
                data_frame[name][n] = data_frame[name][p]

            if np.isnan(data_frame[name][p]) and np.isnan(data_frame[name][n]):
                data_frame[name][i] = 0.0
            else:
                data_frame[name][i] = round(random.uniform(data_frame[name][p], data_frame[name][n]), 1)

    return data_frame


data = pd.read_csv('nocd.csv', na_values='')

data.drop(['STATION', 'NAME'], axis=1, inplace=True)

data = delete_passes(data, 'PRCP')
data = delete_passes(data, 'TAVG')
data = delete_passes(data, 'TMAX')
data = delete_passes(data, 'TMIN')

print(data)

# for i in range(len(data)):
#     if np.isnan(data['PRCP'][i]):
#
#         p = i
#         previous = False
#
#         n = i
#         following = False
#
#         while not previous:
#             p -= 1
#             if not np.isnan(data['PRCP'][p]):
#                 previous = True
#
#         while not following:
#             n += 1
#             if not np.isnan(data['PRCP'][n]):
#                 following = True
#
#         data['PRCP'][i] = round(random.uniform(data['PRCP'][p], data['PRCP'][n]), 1)

