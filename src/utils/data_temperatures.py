import pandas as pd

from utils.plot import plot_time_series


if __name__ == '__main__':
    df = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')
    series = pd.Series(data=df['Temp'].values, index=df['Date'].values)
    plot_time_series(
        'data/bench/img/temperatures.png',
        series.index,
        series.values,
        'Minimum Daily Temperatures')
    series.to_csv('data/bench/temperatures.csv')
