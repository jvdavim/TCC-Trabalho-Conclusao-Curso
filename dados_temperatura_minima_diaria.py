# import pandas as pd

# if __name__ == '__main__':
#     df = pd.read_csv('')
#     series = pd.Series(data=df['Temp'].values, index=df['Date'].values)
#     plot_time_series(
#         'data/bench/img/temperatures.png',
#         series.index,
#         series.values,
#         'Minimum Daily Temperatures')
#     series.to_csv('data/bench/temperatures.csv')

import os

import pandas as pd

if __name__ == '__main__':
    DATA_URL = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv'
    DATA_DIR = os.path.join('data', 'temperatura_minima_diaria')

    df = pd.read_csv(DATA_URL, names=['data', 'temperatura_minima_diaria'], skiprows=1)
    df['data'] = pd.to_datetime(df['data'])

    os.makedirs(DATA_DIR, exist_ok=True)

    df[['data', 'temperatura_minima_diaria']].to_csv(os.path.join(DATA_DIR, 'temperaturas.csv'),
                                                     index=False,
                                                     sep=';',
                                                     date_format='%Y-%m-%d')
