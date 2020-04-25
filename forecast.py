import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def prepare_data(data_names):
    for name in data_names:
        datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Объём', 'Изм. %'], axis=1))
    name = "USD_RUB_rates"
    datasets.append(pd.read_csv("datasets/" + name + ".csv").rename(columns = {'Цена': name}).drop(['Откр.', 'Макс.', 'Мин.', 'Изм. %'], axis=1))
    for i in range(len(datasets)):
        clmn = datasets[i].columns.tolist()
        start_date_index = list(datasets[i]['Дата']).index('01.02.2010')
        end_date_index = list(datasets[i]['Дата']).index('03.02.2014')
        data = {clmn[0]: datasets[i][clmn[0]][end_date_index : start_date_index + 1], clmn[1]: datasets[i][clmn[1]][end_date_index : start_date_index + 1]}
        datasets[i] = pd.DataFrame(data)
    data_ready = datasets[0].copy()
    for i in range(1, len(datasets)):
        data_ready = pd.merge(data_ready, datasets[i], how = 'inner', on = 'Дата')
    data_ready = data_ready.iloc[::-1]
    data_ready.index = range(len(data_ready['Дата']))
    names = list(data_ready.columns)[1 : ]
    #print(list(data_ready.columns)[1 : ])
    for name in names:
        data_ready[name] = data_ready[name].str.replace('.', '')
        data_ready[name] = data_ready[name].str.replace(',', '.')
        print(list(data_ready.columns)[1 : ].index(name))
    #print(data_ready.head())
    data_ready.to_csv("ALL_DATA.csv")
    #print(data_ready.head())
    return data_ready

def visualise_data(data):
    fig = plt.figure(figsize = (19, 8), num = 'Datasets visualisation')
    ax = []
    #fig, ((ax1 ,ax2, ax3, ax4, ax5), (ax6 ,ax7, ax8, ax9, ax10)) = plt.subplots(nrows = 2, ncols = 5, figsize = (4, 8))
    #fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (4, 8))
    #iter = 0
    #ax.append(ax1)
    #ax.append(ax2)
    #ax.append(ax3)
    #ax.append(ax4)
    #x.append(ax5)
    #x.append(ax6)
    #x.append(ax7)
    #ax.append(ax8)
    #ax.append(ax9)
    #ax.append(ax10)
    #print(len(ax))
    #rint(len(list(data.columns)))
    names = list(data.columns)[1 : ]
    print(names)
    print(data.head())
    #for iter in range(len(ax)):
    #    ax[iter].plot(data['Дата'], data[names[iter + 1]])
    #    ax[iter].set_title(names[iter + 1])
    #     iter += 1
    for name in names:
        ax.append(fig.add_subplot(2, 5, names.index(name) + 1, title = name))
        sns.lineplot(data = pd.to_numeric(data[name]), ax = ax[names.index(name)]) # через pd.reaplace()

    plt.show()
    fig.savefig("Visualisation.png")



# INDEXES: DXY, Dow Jones, FTSE 100, MSCI ACWI, мб S&P + русские

#M A I N
datasets = []
names = ["Brent_prices", "ACWI_indexes", "Dow_Jones_indexes",
         "DXY_indexes", "FTSE_100_indexes", "MOEX_indexes", "RTS_indexes", "SPX_indexes"]

#all_data = prepare_data(names)
all_data = pd.read_csv('ALL_DATA.csv', index_col = 0)
#print(all_data.head())
#all_data.set_index('Дата')
#visualise_data(all_data)
