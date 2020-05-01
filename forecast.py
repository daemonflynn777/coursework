import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor

def prepare_data(data_names):
    for name in data_names:
        datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм. %'], axis=1))
    name = "USD_RUB_rates"
    datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Изм. %'], axis=1))
    for i in range(len(datasets)):
        clmn = datasets[i].columns.tolist()
        start_date_index = list(datasets[i]['Дата']).index('01.10.2010') # previous date 01.02.2010
        end_date_index = list(datasets[i]['Дата']).index('01.10.2014') # previous date 01.02.2014
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
    data_ready.to_csv("ALL_DATA.csv")
    return data_ready

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
    fig = plt.figure(figsize = (19, 8), num = 'Datasets correlation')
    ax = []
    names = list(data.columns)[1 : -1]
    for name in names:
        ax.append(fig.add_subplot(2, 5, names.index(name) + 1, title = name))
        sns.lineplot(data = data[name], ax = ax[names.index(name)])

def data_train_test(df):
    df.drop(['Дата'], axis = 1, inplace = True)
    df_x = df.to_numpy()[ : , : len(list(df.columns)) - 1]
    df_y = df.to_numpy()[ : , -1 : ].reshape(1, df.shape[0])
    return df_x, df_y[0]



# INDEXES: DXY, Dow Jones, FTSE 100, MSCI ACWI, мб S&P + русские

#M A I N
datasets = []
names = ["Brent_prices", "ACWI_indexes", "Dow_Jones_indexes",
         "DXY_indexes", "FTSE_100_indexes", "MOEX_indexes", "RTS_indexes", "SPX_indexes"]

all_data = prepare_data(names)
#all_data = pd.read_csv('ALL_DATA.csv', index_col = 0)
all_data['Brent_prices'] = pd.Series(np.ones(len(all_data['Дата'])) / all_data['Brent_prices'].to_numpy().reshape(1, all_data.shape[0])[0])
all_data['MOEX_indexes'] = pd.Series(np.ones(len(all_data['Дата'])) / all_data['MOEX_indexes'].to_numpy().reshape(1, all_data.shape[0])[0])
all_data['RTS_indexes'] = pd.Series(np.ones(len(all_data['Дата'])) / all_data['RTS_indexes'].to_numpy().reshape(1, all_data.shape[0])[0])
visualise_data(all_data)
X, y = data_train_test(all_data)
#print(X)
#print(y)
#all_data.corr().to_csv("CORRELATION.csv")
