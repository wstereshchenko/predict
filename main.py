import datetime
import json
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz

import pydot

def load_data(url, name):
    response = requests.get(url)

    response = response.text
    data = response[9:-1]

    data = json.loads(data)

    normalized_df = pd.io.json.json_normalize(data)
    normalized_df.to_csv(name, index=False)


# load_data('http://localhost:8000/api/ydx/history/', 'ydx_data.csv')
# load_data('http://localhost:8000/api/gism/history/', 'gism_data.csv')
# load_data('http://localhost:8000/api/ydx/history/', 'owm_data.csv')

data = pd.read_csv('ydx_data.csv')

#######
#####################
#####################
#####################

data.drop('date_time', axis=1, inplace=True)

y = data['temp']
x = data.drop('temp', axis=1)

x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=17)

first_tree = DecisionTreeClassifier(random_state=17)

c_v_tree = cross_val_score(first_tree, x_train, y_train, cv=5)

print(np.median(c_v_tree), np.mean(c_v_tree))


first_knn = KNeighborsClassifier()

c_v_knn = cross_val_score(first_knn, x_train, y_train, cv=5)

print(np.median(c_v_knn), np.mean(c_v_knn))

#Настраиваем глубину дерева

tree_params = {'max_depth': np.arange(1, 20), 'max_features': [.5, .7, 1]}
tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1)
tree_grid.fit(x_train, y_train)

print(tree_grid.best_score_, tree_grid.best_params_)

#Настраиваем соседей

knn_params = {'n_neighbors': range(1, 50)}
knn_grid = GridSearchCV(first_knn, knn_params, cv=5, n_jobs=-1)
knn_grid.fit(x_train, y_train)

print(knn_grid.best_score_, knn_grid.best_params_)

#проверим на отложенной выборке

tree_valid_pred = tree_grid.predict(x_valid)

ac_sc_tree = accuracy_score(y_valid, tree_valid_pred)

print(ac_sc_tree)

knn_valid_pred = knn_grid.predict(x_valid)

ac_sc_knn = accuracy_score(y_valid, knn_valid_pred)

print(ac_sc_knn)

#Построим дерево

dot_data = export_graphviz(tree_grid.best_estimator_, out_file='tree.dot', feature_names=x.columns, filled=True)
graph = pydot.graph_from_dot_data(dot_data)
graph.write_png('tree.png')


#####################
#####################
#####################


# date_str = data['date_time'].tolist()
# temp = data['temp'].tolist()
# pressure = data['pressure'].tolist()
# humidity = data['humidity'].tolist()
# date = []

# for _ in date_str:
#     d = datetime.datetime.strptime(_, "%Y-%m-%dT%H:%M:%S.%fZ")
#     d.date()
#     d = datetime.datetime(d.year, d.month, d.day)
#     date.append(int(d.timestamp()))
#
# df = pd.DataFrame({
#     'date': pd.Series(data=date),
#     'temperature': pd.Series(data=temp),
#     'pressure': pd.Series(data=pressure),
#     'humidity': pd.Series(data=humidity)
# })
#
#
# date_counts = df.groupby(['date'])
# present = date_counts.size().to_frame(name='counts')\
#     .join(date_counts.agg({'temperature': 'max'}).rename(columns={'temperature': 'temp_max'}))\
#     .join(date_counts.agg({'pressure': 'median'}).rename(columns={'pressure': 'pressure_median'}))\
#     .join(date_counts.agg({'humidity': 'median'}).rename(columns={'humidity': 'humidity_median'})).reset_index()
#
# average = date_counts.size().to_frame(name='counts')\
#     .join(date_counts.agg({'temperature': 'max'}).rename(columns={'temperature': 'temp_max'}))
#
# average = list(average.columns)
#
# temp_1 = present['temp_max'].tolist()
# temp_1.insert(0, temp_1[0])
# temp_1 = temp_1[:-1]
# temp_2 = temp_1.copy()
# temp_2.insert(0, temp_1[0])
# temp_2 = temp_2[:-1]
# present['temp_max_1'] = temp_1
# present['temp_max_2'] = temp_2
#
#
# labels = np.array(present['temp_max'])
# features = present.drop('temp_max', axis=1)
# feature_list = list(features.columns)
# features = np.array(features)
#
# train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.25,
#                                                                             random_state=42)
#
# print('Training Features Shape:', train_features.shape)
# print('Training Labels Shape:', train_labels.shape)
# print('Testing Features Shape:', test_features.shape)
# print('Testing Labels Shape:', test_labels.shape)
#
# baseline_preds = test_features[:, average.index('temp_max')]
# baseline_errors = abs(baseline_preds - test_labels)
# print('Average baseline error: ', round(np.mean(baseline_errors), 2), 'degrees.')
#
# rf = RandomForestRegressor(n_estimators=1000, random_state=42)
#
# rf.fit(train_features, train_labels)
#
# predictions = rf.predict(test_features)
# errors = abs(predictions - test_labels)
#
# print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
#
# mape = 100 * (errors / test_labels)
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')


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
