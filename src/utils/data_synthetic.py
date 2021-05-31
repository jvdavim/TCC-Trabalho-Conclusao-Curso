from utils.plot import plot_time_series

import pandas as pd
import numpy as np


if __name__ == '__main__':
    # Trend
    time = np.arange(100)
    values = time * 0.4
    plot_time_series('data/synthetic/img/upward_trend.png', time, values, 'Upward Trend')
    pd.Series(data=values, index=time).to_csv('data/synthetic/upward_trend.csv')

    # Seasonality
    time = np.arange(50)
    values = np.where(time < 10, time**3, (time - 9)**2)  # Repeat the pattern 5 times
    seasonal = []
    for i in range(5):
        for j in range(50):
            seasonal.append(values[j])  # Plot
    time_seasonal = np.arange(250)
    plot_time_series('data/synthetic/img/seasonality.png', time_seasonal, seasonal, title='Seasonality')
    pd.Series(data=seasonal, index=time_seasonal).to_csv('data/synthetic/seasonality.csv')

    # Seasonality with noise
    noise = np.random.randn(250) * 100
    seasonal += noise
    time_seasonal = np.arange(250)
    plot_time_series('data/synthetic/img/seasonality_with_noise.png',
                     time_seasonal, seasonal, title='Seasonality with Noise')
    pd.Series(data=seasonal, index=time_seasonal).to_csv('data/synthetic/seasonal_with_noise.csv')

    # Multiple patterns
    seasonal_upward = seasonal + np.arange(250) * 10
    time_seasonal = np.arange(250)
    plot_time_series(
        'data/synthetic/img/multiple_patterns.png',
        time_seasonal,
        seasonal_upward,
        title='Seasonality + Upward Trend + Noise')
    pd.Series(data=seasonal_upward, index=time_seasonal).to_csv('data/synthetic/multiple_patterns.csv')

    # White noise
    time = np.arange(200)
    values = np.random.randn(200) * 100
    plot_time_series('data/synthetic/img/white_noise.png', time, values, title='White Noise')
    pd.Series(data=seasonal_upward, index=time_seasonal).to_csv('data/synthetic/white_noise.csv')

    # Non-stationary
    big_event = np.zeros(250)
    big_event[-50:] = np.arange(50) * -50
    non_stationary = seasonal_upward + big_event
    time_seasonal = np.arange(250)
    plot_time_series(
        'data/synthetic/img/non_stationary.png',
        time_seasonal,
        non_stationary,
        title='Non-stationary Time Series')
    pd.Series(data=non_stationary, index=time_seasonal).to_csv('data/synthetic/non_stationary.csv')
