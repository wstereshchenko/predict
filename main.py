import datetime
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import export_graphviz
import pydot
from sklearn.tree import export_graphviz


def load_data(url, name):
    response = requests.get(url)

    response = response.text
    data = response[9:-1]

    data = json.loads(data)

    normalized_df = pd.io.json.json_normalize(data)
    normalized_df.to_csv(name, index=False)


def predict_1(present, average):
    labels = np.array(present['temp_max'])
    features = present.drop('temp_max', axis=1)
    feature_list = list(features.columns)
    print(feature_list)
    features = np.array(features)

    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
                                                                                random_state=42)

    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

    baseline_preds = test_features[:, feature_list.index('temp_max')]
    baseline_errors = abs(baseline_preds - test_labels)
    print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'degrees.')

    rf = RandomForestRegressor(n_estimators=1000, random_state=42)

    rf.fit(train_features, train_labels)

    predictions = rf.predict(test_features)
    errors = abs(predictions - test_labels)

    print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

    mape = 100 * (errors / test_labels)
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')


load_data('http://localhost:8000/api/ydx/history/', 'ydx_data.csv')
load_data('http://localhost:8000/api/gism/history/', 'gism_data.csv')
load_data('http://localhost:8000/api/ydx/history/', 'owm_data.csv')

#######

data = pd.read_csv('ydx_data.csv')

date_str = data['date_time'].tolist()
temp = data['temp'].tolist()
pressure = data['pressure'].tolist()
humidity = data['humidity'].tolist()
date = []

for _ in date_str:
    d = datetime.datetime.strptime(_, "%Y-%m-%dT%H:%M:%S.%fZ")
    d.date()
    d = datetime.datetime(d.year, d.month, d.day)
    date.append(int(d.timestamp()))

df = pd.DataFrame({
    'date': pd.Series(data=date),
    'temperature': pd.Series(data=temp),
    'pressure': pd.Series(data=pressure),
    'humidity': pd.Series(data=humidity)
})

date_counts = df.groupby(['date'])
present = date_counts.size().to_frame(name='counts')\
    .join(date_counts.agg({'temperature': 'max'}).rename(columns={'temperature': 'temp_max'}))\
    .join(date_counts.agg({'pressure': 'median'}).rename(columns={'pressure': 'pressure_median'}))\
    .join(date_counts.agg({'humidity': 'median'}).rename(columns={'humidity': 'humidity_median'})).reset_index()

average = date_counts.size().to_frame(name='counts')\
    .join(date_counts.agg({'temperature': 'max'}).rename(columns={'temperature': 'temp_max'}))

average = list(average.columns)

temp_1 = present['temp_max'].tolist()
temp_1.insert(0, temp_1[0])
temp_1 = temp_1[:-1]
temp_2 = temp_1.copy()
temp_2.insert(0, temp_1[0])
temp_2 = temp_2[:-1]
present['temp_max_1'] = temp_1
present['temp_max_2'] = temp_2

predict_1(present, average)

#####

######
#
# tree = rf.estimators_[5]
# export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
# (graph, ) = pydot.graph_from_dot_file('tree.dot')
# graph.write_png('tree.png')

# rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
# rf_small.fit(train_features, train_labels)
# tree_small = rf_small.estimators_[5]
# export_graphviz(tree_small, out_file='small_tree.dot', feature_names=feature_list, rounded=True, precision=1)
# (graph, ) = pydot.graph_from_dot_file('small_tree.dot')
# graph.write_png('small_tree.png')

######
# temp = data['temp'].tolist()
# date = data['date_time'].tolist()
#
# temp = pd.Series(data=temp, index=date,)

# plt.style.use('fivethirtyeight')

# dates = data['date_time'].tolist()
# temp = data['temp'].tolist()
# print(dates)
#
# fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
# fig.autofmt_xdate(rotation=45)
#
# ax1.plot(present['date'], present['temp_max_1'])
# ax1.set_xlabel('')
# ax1.set_ylabel('Temperature')
# ax1.set_title('temp_max_1')
#
# ax2.plot(present['date'], present['temp_max_2'])
# ax2.set_xlabel('')
# ax2.set_ylabel('Temperature')
# ax2.set_title('temp_max_2')
#
# ax3.plot(present['date'], present['humidity_median'])
# ax3.set_xlabel('')
# ax3.set_ylabel('Humidity')
# ax3.set_title('humidity')
#
# ax4.plot(present['date'], present['pressure_median'])
# ax4.set_xlabel('')
# ax4.set_ylabel('Pressure')
# ax4.set_title('pressure')
#
# plt.show()
