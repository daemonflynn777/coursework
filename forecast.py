import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret as pc
import xgboost as xgb

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def prepare_data(data_names):
    for name in data_names:
        datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм. %'], axis=1))
    name = "USD_RUB_rates"
    datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Изм. %'], axis=1))
    for i in range(len(datasets)):
        clmn = datasets[i].columns.tolist()
        start_date_index = list(datasets[i]['Дата']).index('01.10.2010') # previous date 01.02.2010
        #mid_date_index = list(datasets[i]['Дата']).index('01.10.2014') # previous date 01.02.2014
        end_date_index = list(datasets[i]['Дата']).index('01.10.2018')
        data = {clmn[0]: datasets[i][clmn[0]][end_date_index : start_date_index + 1], clmn[1]: datasets[i][clmn[1]][end_date_index : start_date_index + 1]}
        datasets[i] = pd.DataFrame(data)
    data_ready = datasets[0].copy()
    for i in range(1, len(datasets)):
        data_ready = pd.merge(data_ready, datasets[i], how = 'inner', on = 'Дата')
    data_ready = data_ready.iloc[::-1]
    data_ready.index = range(len(data_ready['Дата']))
    names = list(data_ready.columns)[1 : ]
    for name in names:
        data_ready[name] = data_ready[name].str.replace('.', '')
        data_ready[name] = pd.to_numeric(data_ready[name].str.replace(',', '.'))
    data_test = data_ready.iloc[list(data_ready['Дата']).index('01.10.2014') : ]
    data_ready = data_ready.iloc[ : list(data_ready['Дата']).index('01.10.2014') + 1]
    data_test.index = range(len(data_test['Дата']))
    data_ready.to_csv("ALL_DATA.csv")
    data_test.to_csv("TEST_DATA.csv")
    return data_ready, data_test

def visualise_data(data):
    fig = plt.figure(figsize = (19, 8), num = 'Datasets visualisation')
    ax = []
    names = list(data.columns)[1 : ]
    print(names)
    print(data.head())
    for name in names:
        ax.append(fig.add_subplot(2, 5, names.index(name) + 1, title = name))
        sns.lineplot(data = data[name], ax = ax[names.index(name)]) # через pd.reaplace()
    plt.show()
    fig.savefig("Visualisation.png")
    fig = plt.figure(figsize = (12 ,9), num = 'Correlation heatmap')
    sns.heatmap(data = data.drop(['Дата'], axis = 1).corr(), annot = True, vmin = -1, vmax = 1, center = 0)
    plt.show()
    fig.savefig("Correlation_heatmap.png")
    data.drop(['Дата'], axis = 1).corr().to_csv("CORRELATION.csv")

def visualise_corr(data):
    fig = plt.figure(figsize = (21, 8), num = 'Datasets correlation')
    ax = []
    names = list(data.columns)[1 : -1]
    print(data.head())
    for name in names:
        ax.append(fig.add_subplot(2, 4, names.index(name) + 1))
        sns.regplot(x = 'USD_RUB_rates', y = name, data = data, ax = ax[names.index(name)])
    plt.subplots_adjust(wspace = 0.3)
    plt.show()
    fig.savefig("Datasets_correlation.png")

def data_train_test(df):
    df.drop(['Дата'], axis = 1, inplace = True)
    df_x = df.to_numpy()[ : , : len(list(df.columns)) - 1]
    df_y = df.to_numpy()[ : , -1 : ].reshape(1, df.shape[0])
    return df_x, df_y[0]

def xgb_forecasting(data_x, data_y):
    dtrain = xgb.DMatrix(data_x, label = data_y)
    xgb_model = xgb.XGBRegressor(learning_rate = 0.1, nthread = -1, random_state = 0)
    cv_gen = ShuffleSplit(n_splits = 9, test_size = 0.7, random_state = 0)
    xgb_gs = GridSearchCV(
             xgb_model,
             {
                'n_estimators': [125],
                'max_depth': [3],
                'booster': ['gbtree', 'gblinear'],
                'gamma': np.linspace(0.00005, 0.0001, num = 5),
                'reg_alpha': np.linspace(0.0001, 0.001, num = 3),
                'reg_lambda': np.linspace(0.0001, 0.001, num = 3)
             },
             scoring = 'r2', # neg_mean_squared_error
             n_jobs = -1,
             cv = cv_gen
             )
    xgb_gs.fit(data_x[1 : ], data_y[1 : ])
    print(xgb_gs.best_params_)
    print("r2 score",xgb_gs.best_score_)
    y_pred = xgb_gs.best_estimator_.predict(data_x[ : 1])
    #y_pred = xgb_gs.best_estimator_.predict(X_test[300].reshape(1, X_test.shape[1]))
    print("Predicted", y_pred)
    print("Real", data_y[ : 1])
    #print("Real", y_test[300])

def sklearn_forecasting(mdls, prms, data_x, data_y):
    data_y = data_y.astype('float')
    print(data_y)
    bst_params = []
    bst_score = []
    bst_estimator = []
    cv_gen = ShuffleSplit(n_splits = 9, test_size = 0.7, random_state = 0)
    #for i in range(len(mdls)):
    model_gs = GridSearchCV(mdls[4], prms[4], scoring = 'r2', n_jobs = -1, cv = cv_gen)
    model_gs.fit(data_x[ : -1], data_y[ : -1])
    bst_params.append(model_gs.best_params_)
    bst_score.append(model_gs.best_score_)
    bst_estimator.append(model_gs.best_estimator_)
    print(bst_score)

    return 0




# INDEXES: DXY, Dow Jones, FTSE 100, MSCI ACWI, мб S&P + русские

#M A I N
datasets = []
names = ["Brent_prices", "ACWI_indexes", "Dow_Jones_indexes",
         "DXY_indexes", "FTSE_100_indexes", "MOEX_indexes", "RTS_indexes", "SPX_indexes"]

all_data, test_data = prepare_data(names)
#all_data = pd.read_csv('ALL_DATA.csv', index_col = 0)
#test_data = pd.read_csv('TEST_DATA.csv', index_col = 0)

all_data['Brent_prices'] = pd.Series(np.ones(len(all_data['Дата'])) / all_data['Brent_prices'].to_numpy().reshape(1, all_data.shape[0])[0])
all_data['MOEX_indexes'] = pd.Series(np.ones(len(all_data['Дата'])) / all_data['MOEX_indexes'].to_numpy().reshape(1, all_data.shape[0])[0])
all_data['RTS_indexes'] = pd.Series(np.ones(len(all_data['Дата'])) / all_data['RTS_indexes'].to_numpy().reshape(1, all_data.shape[0])[0])
test_data['Brent_prices'] = pd.Series(np.ones(len(test_data['Дата'])) / test_data['Brent_prices'].to_numpy().reshape(1, test_data.shape[0])[0])
test_data['MOEX_indexes'] = pd.Series(np.ones(len(test_data['Дата'])) / test_data['MOEX_indexes'].to_numpy().reshape(1, test_data.shape[0])[0])
test_data['RTS_indexes'] = pd.Series(np.ones(len(test_data['Дата'])) / test_data['RTS_indexes'].to_numpy().reshape(1, test_data.shape[0])[0])

#visualise_data(all_data)
#visualise_corr(all_data)
#print(test_data['RTS_indexes'])

X, y = data_train_test(all_data)
X_test, y_test = data_train_test(test_data)

#xgb_forecasting(X, y)

models = []
models.append(LinearRegression(copy_X = True, n_jobs = -1))
#models.append(LogisticRegression(n_jobs = -1, class_weight = 'balanced', random_state = 0))
models.append(Ridge(copy_X = True, random_state = 0))
models.append(Lasso(copy_X = True, random_state = 0))
models.append(KNeighborsRegressor(n_jobs = -1))
models.append(MLPRegressor(random_state = 0))


#'l1_ratio' : np.linspace(0.0, 1.0, num = 5)
params = []
params.append({'fit_intercept' : [True, False], 'normalize' : [True, False]}) # params for Linear Regression
#params.append({'penalty' : ['l2'], 'dual' : [False], 'tol' : np.linspace(0.00001, 0.0001, num = 5), 'C' : np.linspace(0.1, 2.0, num = 10),
#               'fit_intercept' : [False], 'solver' : ['newton-cg', 'lbfgs', 'sag'], 'max_iter': [100, 125, 150, 175, 200], 'multi_class' : ['auto', 'ovr', 'multinominal']})
params.append({'alpha' : np.linspace(0.1, 2.0, num = 10), 'fit_intercept' : [True, False], 'normalize' : [True, False], 'tol' : np.linspace(0.00001, 0.0001, num = 5),
               'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparce_cg', 'sag', 'saga']})
params.append({'alpha' : np.linspace(1.0, 5.0, num = 10), 'fit_intercept' : [True, False], 'normalize' : [True, False], 'precompute' : [True, False], 'tol' : np.linspace(0.00001, 0.0001, num = 5)})
params.append({'n_neighbors' : [5, 10, 15, 20], 'weights' : ['uniform', 'distance'], 'algorithm' : ['ball_tree', 'kd_tree', 'brute'], 'leaf_size' : [30, 45, 60, 75, 90], 'p' : np.linspace(1, 5, num = 6)})

params.append({'activation' : ['logistic'], 'solver' : ['lbfgs'], 'alpha' : [0.00005, 0.0001, 0.0002], 'tol' : np.linspace(0.00001, 0.0001, num = 5),
               'early_stopping' : [True], 'validation_fraction' : np.linspace(0.1, 0.3, num = 5)})

sklearn_forecasting(models, params, X, y)

#print(X)
#print(y)
#all_data.corr().to_csv("CORRELATION.csv")
