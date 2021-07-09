import datetime
import os

import numpy as np
import pandas as pd

if __name__ == '__main__':
    DATA_DIR = os.path.join('data', 'sinteticos')

    # Trend
    time = np.arange(100)
    values = time * 0.4

    # Random seasonal pattern
    time = np.arange(50)
    values = np.where(time < 10, time**3, (time - 9)**2)
    seasonal = []
    for i in range(5):
        for j in range(50):
            seasonal.append(values[j])

    # Seasonality with noise
    noise = np.random.randn(250) * 100
    seasonal += noise

    # Multiple patterns
    seasonal_upward = seasonal + np.arange(250) * 10

    # White noise
    values = np.random.randn(200) * 100

    # Non-stationary
    big_event = np.zeros(250)
    big_event[-50:] = np.arange(50) * -50
    non_stationary = seasonal_upward + big_event
    non_stationary = [round(x, 1) for x in non_stationary]
    dates = pd.date_range(start=datetime.date(2020, 12, 14) - datetime.timedelta(days=len(non_stationary)),
                          end=datetime.date(2020, 12, 13))

    os.makedirs(DATA_DIR, exist_ok=True)

    pd.DataFrame(data=zip(dates, non_stationary), columns=['data',
                                                           'valor']).to_csv(os.path.join(DATA_DIR, 'sinteticos.csv'),
                                                                            index=False,
                                                                            sep=';',
                                                                            date_format='%Y-%m-%d')
