import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import median

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils
from sklearn.svm import SVC
from sklearn import ensemble

from pandas.plotting import scatter_matrix

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
                data_frame[name][p] = data_frame[name][n] + random.randint(-3, 3)

            if np.isnan(data_frame[name][n]):
                data_frame[name][n] = data_frame[name][p] + random.randint(-3, 3)

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

temp_max_y = list(data['TMAX'])
temp_min_y = list(data['TMIN'])
temp_avg_y = list(data['TAVG'])
temp_max_t = list(data['TMAX'])

month_day = list(data['DATE'])
season = list(data['DATE'])

for i in range(len(month_day)):
    month_day[i] = month_day[i][5:]
    if month_day[i][0:2] == '12' or month_day[i][0:2] == '01' or month_day[i][0:2] == '02':
        season[i] = 'Winter'

    elif month_day[i][0:2] == '03' or month_day[i][0:2] == '04' or month_day[i][0:2] == '05':
        season[i] = 'Spring'

    elif month_day[i][0:2] == '06' or month_day[i][0:2] == '07' or month_day[i][0:2] == '08':
        season[i] = 'Summer'

    else:
        season[i] = 'Autumn'


temp_avg_y[1:len(temp_avg_y)-1] = temp_avg_y
temp_max_y[1:len(temp_max_y)-1] = temp_max_y
temp_min_y[1:len(temp_min_y)-1] = temp_min_y
temp_max_t.pop(0)

################

data['TAVG'] = pd.Series(temp_avg_y)
data['TMAX'] = pd.Series(temp_max_y)
data['TMIN'] = pd.Series(temp_min_y)
data['MD'] = pd.Series(month_day)
data['SEASON'] = pd.Series(season)


#######
# data['TMAXT'] = pd.Series(temp_max_t) не подходит

anomaly = []
for i in range(1, len(data['TAVG'])):
    anomaly.append(abs(round(data['TMAX'][i] - data['TMAX'][i-1])))

# anomaly = pd.Series(anomaly)

print(sorted(anomaly))

dct = {}

for i in anomaly:
    if i in dct:
        dct[i] += 1
    else:
        dct[i] = 1

for i in sorted(dct):
    print("'%d':%d" % (i, dct[i]))

for i in range(len(anomaly)):
    if anomaly[i] > 5:
        anomaly[i] = True
    else:
        anomaly[i] = False

data['ANOMALY'] = pd.Series(anomaly)

#######

data.drop(['DATE'], axis=1, inplace=True)

categorical_columns = [c for c in data.columns if data[c].dtype.name == 'object']
numerical_columns = [c for c in data.columns if data[c].dtype.name != 'object']

# scatter_matrix(data, alpha=0.05, figsize=(10, 10))
# plt.show()

data = data.dropna(axis=0)

print(data.describe())

data_describe = data.describe(include=[object])
binary_columns = [c for c in categorical_columns if data_describe[c]['unique'] == 2]
nonbinary_columns = [c for c in categorical_columns if data_describe[c]['unique'] > 2]
print(binary_columns, nonbinary_columns)

for c in binary_columns:
    top = data_describe[c]['top']
    top_items = data[c] == top
    data.loc[top_items, c] = 0
    data.loc[np.logical_not(top_items), c] = 1

data_nonbinary = pd.get_dummies(data[nonbinary_columns])
print(data_nonbinary.columns)

data_numerical = data[numerical_columns]
data_numerical = (data_numerical - data_numerical.mean()) / data_numerical.std()
print(data_numerical.describe())

data = pd.concat((data_numerical, data[binary_columns], data_nonbinary), axis=1)
data = pd.DataFrame(data, dtype=float)
print(data.shape)
print(data.columns)


X = data.drop(('ANOMALY'), axis=1)
y = data['ANOMALY']
feature_names = X.columns
print(feature_names)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)

N_train, _ = X_train.shape
N_test,  _ = X_test.shape
print(N_train, N_test)


print(60 * '=')
print('kNN')

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_train_predict = knn.predict(X_train)
y_test_predict = knn.predict(X_test)

err_train = np.mean(y_train != y_train_predict)
err_test = np.mean(y_test != y_test_predict)
print(err_train, err_test)

n_neighbors_array = [1, 3, 5, 7, 10, 15]
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid={'n_neighbors': n_neighbors_array})
grid.fit(X_train, y_train)

best_cv_err = 1 - grid.best_score_
best_n_neighbors = grid.best_estimator_.n_neighbors
print(best_cv_err, best_n_neighbors)


knn = KNeighborsClassifier(n_neighbors=best_n_neighbors)
knn.fit(X_train, y_train)

err_train = np.mean(y_train != knn.predict(X_train))
err_test = np.mean(y_test != knn.predict(X_test))
print(err_train, err_test)

print(60 * '=')

print('SVC')

svc = SVC()
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))
print(err_train, err_test)

print('rbf')

C_array = np.logspace(-3, 3, num=7)
gamma_array = np.logspace(-5, 2, num=8)
svc = SVC(kernel='rbf')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array})
grid.fit(X_train, y_train)
print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)
print('best gamma  = ', grid.best_estimator_.gamma)

svc = SVC(kernel='rbf', C=grid.best_estimator_.C, gamma=grid.best_estimator_.gamma)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))
print(err_train, err_test)

print(60 * '-')
print('linear')

C_array = np.logspace(-3, 3, num=7)
svc = SVC(kernel='linear')
grid = GridSearchCV(svc, param_grid={'C': C_array})
grid.fit(X_train, y_train)
print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)

svc = SVC(kernel='linear', C=grid.best_estimator_.C)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))
print(err_train, err_test)

print(60 * '-')
print('poly')

C_array = np.logspace(-5, 2, num=8)
gamma_array = np.logspace(-5, 2, num=8)
degree_array = [2, 3, 4]
svc = SVC(kernel='poly')
grid = GridSearchCV(svc, param_grid={'C': C_array, 'gamma': gamma_array, 'degree': degree_array})
grid.fit(X_train, y_train)
print('CV error    = ', 1 - grid.best_score_)
print('best C      = ', grid.best_estimator_.C)
print('best gamma  = ', grid.best_estimator_.gamma)
print('best degree = ', grid.best_estimator_.degree)

svc = SVC(kernel='poly', C=grid.best_estimator_.C,
          gamma=grid.best_estimator_.gamma, degree=grid.best_estimator_.degree)
svc.fit(X_train, y_train)

err_train = np.mean(y_train != svc.predict(X_train))
err_test = np.mean(y_test != svc.predict(X_test))
print(err_train, err_test)

print(60 * '=')
print('Random Forest')

rf = ensemble.RandomForestClassifier(n_estimators=100, random_state=11)
rf.fit(X_train, y_train)

err_train = np.mean(y_train != rf.predict(X_train))
err_test = np.mean(y_test != rf.predict(X_test))
print(err_train, err_test)

print(30 * '//')
#
# importances = rf.feature_importances_
# indices = np.argsort(importances)[::-1]
#
# print("Feature importances:")
# for f, idx in enumerate(indices):
#     print("{:2d}. feature '{:5s}' ({:.4f})".format(f + 1, feature_names[idx], importances[idx]))

####################

# best_features = indices[:8]
# best_features_names = feature_names[best_features]
# print(best_features_names)

################


# data.drop(data[-1:], inplace=True)
#
# data.dropna(inplace=True)
# data.drop(['DATE'], axis=1, inplace=True)
#
# y = data['TMAXT']
# x = data.drop('TMAXT', axis=1)
#
#
# x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.3, random_state=17)

# lab_enc = preprocessing.LabelEncoder()
# y_train = lab_enc.fit_transform(y_train)
# print(y_train)
# print(utils.multiclass.type_of_target(y_train))
# print(utils.multiclass.type_of_target(y_train.astype('int')))
# print(utils.multiclass.type_of_target(y_train))

# first_tree = DecisionTreeClassifier(random_state=17)
#
# c_v_tree = cross_val_score(first_tree, x_train, y_train, cv=5)
#
# print(np.median(c_v_tree), np.mean(c_v_tree))
#
# first_knn = KNeighborsClassifier()
#
# c_v_knn = cross_val_score(first_knn, x_train, y_train, cv=5)
#
# print(np.median(c_v_knn), np.mean(c_v_knn))
#
# #Настраиваем глубину дерева
#
# tree_params = {'max_depth': np.arange(1, 20), 'max_features': [.5, .7, 1]}
# tree_grid = GridSearchCV(first_tree, tree_params, cv=5, n_jobs=-1)
# tree_grid.fit(x_train, y_train)
#
# print(tree_grid.best_score_, tree_grid.best_params_)
#
# #Настраиваем соседей
#
# knn_params = {'n_neighbors': range(1, 50)}
# knn_grid = GridSearchCV(first_knn, knn_params, cv=5, n_jobs=-1)
# knn_grid.fit(x_train, y_train)
#
# print(knn_grid.best_score_, knn_grid.best_params_)
#
# #проверим на отложенной выборке
#
# tree_valid_pred = tree_grid.predict(x_valid)
#
# ac_sc_tree = accuracy_score(y_valid, tree_valid_pred)
#
# print(ac_sc_tree)
#
# knn_valid_pred = knn_grid.predict(x_valid)
#
# ac_sc_knn = accuracy_score(y_valid, knn_valid_pred)
#
# print(ac_sc_knn)


