import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret as pc
import xgboost as xgb

from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from itertools import product
from functools import reduce
from sklearn.exceptions import ConvergenceWarning, FitFailedWarning
import warnings

def prepare_data(data_names):
    for name in data_names:
        datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм. %'], axis=1))
    name = "USD_RUB_rates"
    datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Изм. %'], axis=1))
    for i in range(len(datasets)):
        clmn = datasets[i].columns.tolist()
        start_date_index = list(datasets[i]['Дата']).index('01.10.2010')
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
        sns.lineplot(data = data[name], ax = ax[names.index(name)])
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

def xgb_forecasting(data_x, data_y, data_features, logfile):
    dtrain = xgb.DMatrix(data_x, label = data_y)
    xgb_model = xgb.XGBRegressor(learning_rate = 0.1, verbosity = 0, nthread = -1, random_state = 0)
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
             scoring = 'r2',
             n_jobs = -1,
             cv = cv_gen
             )
    xgb_gs.fit(data_x, data_y)
    print(xgb_gs.best_params_)
    print("r2 score",xgb_gs.best_score_)
    y_pred = xgb_gs.best_estimator_.predict(data_features)
    print("\nXGBoost forecasted USD/RUB rate:", y_pred)
    logfile.write("\nXGBoost forecasted USD/RUB rate: ")
    logfile.write(str(y_pred))

def feature_data(features, mdls, prms, logfile):
    features = np.transpose(features)
    forecasted = []
    for feature in features:
        #fig = plt.figure(figsize = (8, 8), num = 'ACF and PACF')
        #ax = []
        #ax.append(fig.add_subplot(2, 1, 1))
        #plot_acf(feature, ax = ax[0])
        #ax.append(fig.add_subplot(2, 1, 2))
        #plot_pacf(feature, ax = ax[1])
        #plt.show()
        #plt.show()
        #pacf_coef = int(input("Enter PACF coefficent: ")) # AR part of time series, parameter "p"

        time_delta = 14
        X = []
        y = []
        plt.show()
        for i in range(len(feature) - time_delta):
            X.append(feature[i : i + time_delta])
            y.append(feature[i + time_delta])
        X = np.array(X)
        print(X)
        y = np.array(y)
        for_forecast = np.array(feature[-time_delta : ]).reshape(1, time_delta)
        forecasted.append(sklearn_forecasting(mdls, prms, X, y, for_forecast, logfile))
    logfile.write("\nMean SKLearn forecast for all features:\n")
    for frcst in forecasted:
        logfile.write(str(frcst) + " ")
    logfile.write("\n")
    return forecasted

def sklearn_forecasting(mdls, prms, data_x, data_y, data_feature, logfile):
    data_y = data_y.astype('float')
    bst_params = []
    bst_score = []
    bst_estimator = []
    bst_forecast = []
    print(data_feature)
    cv_gen = ShuffleSplit(n_splits = 9, test_size = 0.7, random_state = 0)
    for i in range(len(mdls)):
        model_gs = GridSearchCV(mdls[i], prms[i], scoring = 'r2', n_jobs = -1, cv = cv_gen)
        model_gs.fit(data_x, data_y)
        bst_params.append(model_gs.best_params_)
        bst_score.append(model_gs.best_score_)
        bst_estimator.append(model_gs.best_estimator_)
        bst_forecast.append(model_gs.best_estimator_.predict(data_feature)[0])
    print("\nSKLearn models scores and forecasts:")
    print(bst_score)
    print(bst_forecast)
    print("SKLearn mean forecast across all models:", reduce(lambda a, b: a + b, bst_forecast)/len(bst_forecast))

    logfile.write("\nSKLearn models scores and forecasts:\n")
    for bst_s in bst_score:
        logfile.write(str(bst_s) + " ")
    logfile.write("\n")
    for bst_f in bst_forecast:
        logfile.write(str(bst_f) + " ")
    logfile.write("\n")
    logfile.write("SKLearn mean forecast across all models: ")
    logfile.write(str(reduce(lambda a, b: a + b, bst_forecast)/len(bst_forecast)))
    logfile.write("\n")
    return reduce(lambda a, b: a + b, bst_forecast)/len(bst_forecast)

def time_series_diff(series):
    return np.array([series[i + 1] - series[i] for i in range(len(series) - 1)])

def arima_forecasting(features):
    warnings.simplefilter('ignore')
    forecasted_features = []
    features = np.transpose(features)
    for feature in features:
        feature_len = len(feature)
        cv_split = int(feature_len*0.85)
        feature_train = feature[ : cv_split]
        feature_test = feature[cv_split : ]
        feature_forecast = feature_train

        #adf_res = adfuller(feature, regression = 'ct')
        #adf_stat = adf_res[1]
        #adf_crit_val = list(adf_res[4].values())[1]
        #adf_crit_val = adf_res[4]["5%"]
        #print(adf_crit_val)

        adf_stat, adf_crit_val = adfuller(feature, regression = 'ctt')[0], adfuller(feature, regression = 'ctt')[4]["5%"]
        int_degree = 0
        while adf_stat >= adf_crit_val:
            print("\n", adf_stat, adf_crit_val)

            #fig = plt.figure(figsize = (8, 4))
            #ft = plt.plot(feature)
            #lt.legend()
            #lt.grid(linewidth = 1)
            #plt.title('Feature', fontsize = 'xx-large')
            #lt.show()

            feature = np.diff(feature)
            adf_stat, adf_crit_val = adfuller(feature, regression = 'ctt')[0], adfuller(feature, regression = 'ctt')[4]["5%"]
            int_degree += 1
        print("Time series is stationary with d = ", int_degree)
        fig = plt.figure(figsize = (8, 8), num = 'ACF and PACF')
        ax = []
        ax.append(fig.add_subplot(2, 1, 1))
        plot_acf(feature, ax = ax[0])
        ax.append(fig.add_subplot(2, 1, 2))
        plot_pacf(feature, ax = ax[1])
        plt.show()
        acf_coef = int(input("Enter ACF coefficent: ")) # MA part of time series, parameter "q"
        pacf_coef = int(input("Enter PACF coefficent: ")) # AR part of time series, parameter "p"
        parameters = product(range(pacf_coef + 1), range(acf_coef + 1))
        parameters_list = list(parameters)
        model_score = model_score_best = q_best = p_best = 0 # Maybe use AIC later
        forecasted_best = list(np.zeros(len(feature_test)))
        for params in parameters_list:
            print("Testing ARIMA (%d,%d,%d)" % (params[0], int_degree, params[1]))
            forecasted = []
            for iter in range(len(feature_test)):
                model = ARIMA(feature_train, order = (params[0], int_degree, params[1]))
                model_fit = model.fit(disp = 0)
                #forecast_step = model_fit.forecast(len(feature_test) - iter + 1)[0][0]
                forecast_step = model_fit.forecast()[0]
                forecasted.append(forecast_step)
                feature_train = np.append(feature_train, feature_test[iter])
            model_score = r2_score(feature_test, forecasted)
            print(model_score)
            if model_score > model_score_best:
                model_score_best = model_score
                p_best = params[0]
                q_best = params[1]
                forecasted_best = forecasted
        print("The best model by r2_score is ARIMA(%d,%d,%d) with r2_score of %f" % (p_best, int_degree, q_best, model_score_best))

        fig = plt.figure(figsize = (8, 5), num = 'Predicted VS Real')
        plt.plot(feature_test, color = 'orange', label = 'Real')
        plt.plot(forecasted, color = 'purple', label = 'Predicted')
        plt.legend()
        plt.grid(linewidth = 1)
        plt.title('Predicted VS Real', fontsize = 'xx-large')
        plt.show()

        model = ARIMA(feature_forecast, order = (p_best, int_degree, q_best))
        model_fit = model.fit(disp = 0)
        forecast = model_fit.forecast(steps = 2)
        forecasted_features.append(forecast[0][1])
    print("\nAll features have been forecasted")
    return forecasted_features


# INDEXES: DXY, Dow Jones, FTSE 100, MSCI ACWI, мб S&P + русские

#M A I N
datasets = []
names = ["Brent_prices", "ACWI_indexes", "Dow_Jones_indexes",
         "DXY_indexes", "FTSE_100_indexes", "MOEX_indexes", "RTS_indexes", "SPX_indexes"]

#all_data, test_data = prepare_data(names)
all_data = pd.read_csv('ALL_DATA.csv', index_col = 0)
test_data = pd.read_csv('TEST_DATA.csv', index_col = 0)

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

X = X[99 : -230, : ]
y = y[99 : -230]

forecast_report = open("Forecast_report.txt", "w")

#forecasted_features_arima = arima_forecasting(X)
forecasted_features_arima = [0.009436234227107878, 51.48247226605893, 15076.90028526149, 80.65615687116073, 6300.875627816384, 0.0007695923111092246, 0.0007730595155224728, 1627.4914738510302]
print("ARIMA forecasted future features:", forecasted_features_arima)
forecast_report.write("ARIMA forecasted future features: ")
for ffa in forecasted_features_arima:
    forecast_report.write(str(ffa) + " ")
forecast_report.write("\n")

models = []
models.append(LinearRegression(copy_X = True, n_jobs = -1))
models.append(Ridge(copy_X = True, random_state = 0))
models.append(Lasso(copy_X = True, random_state = 0))
models.append(KNeighborsRegressor(n_jobs = -1))
models.append(RandomForestRegressor(n_jobs = -1, random_state = 0, verbose = 0))

params = []
params.append({'fit_intercept' : [True, False], 'normalize' : [True, False]}) # params for Linear Regression
params.append({'alpha' : np.linspace(0.1, 2.0, num = 10), 'fit_intercept' : [True, False], 'normalize' : [True, False], 'tol' : np.linspace(0.00001, 0.0001, num = 5),
               'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparce_cg', 'sag', 'saga']})
params.append({'alpha' : np.linspace(1.0, 5.0, num = 10), 'fit_intercept' : [True, False], 'normalize' : [True, False], 'precompute' : [True, False], 'tol' : np.linspace(0.00001, 0.0001, num = 5)})
params.append({'n_neighbors' : [5, 10, 15, 20], 'weights' : ['uniform', 'distance'], 'algorithm' : ['ball_tree', 'kd_tree', 'brute'], 'leaf_size' : [30, 45, 60, 75, 90], 'p' : np.linspace(1, 5, num = 6)})
params.append({'n_estimators' : [100, 125, 150, 175], 'max_depth' : [3, 7, 15], 'max_features' : ['auto', 'sqrt', 'log2']})

forecasted_features_sklearn = feature_data(X, models, params, forecast_report)
print("SKLearn forecasted future features:", forecasted_features_sklearn)
forecast_report.write("SKLearn forecasted future features: ")
for ffa in forecasted_features_sklearn:
    forecast_report.write(str(ffa) + " ")
forecast_report.write("\n")

#ARIMA predicted features:
#sklearn_forecasting(models, params, X, y, np.array(forecasted_features_arima).reshape(1, len(forecasted_features_arima)), forecast_report)

#xgb_forecasting(X, y, forecasted_features_arima, forecast_report)

#SKLearn predicted features
sklearn_forecasting(models, params, X, y, np.array(forecasted_features_sklearn).reshape(1, len(forecasted_features_sklearn)), forecast_report)

xgb_forecasting(X, y, forecasted_features_sklearn, forecast_report)

forecast_report.close()
