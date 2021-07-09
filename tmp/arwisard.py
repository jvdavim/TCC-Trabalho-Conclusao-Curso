import argparse
import itertools
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wisardpkg as wp
from memory_profiler import profile
from tqdm import tqdm

from utils.metrics import *
from utils.plot import plot_observed_vs_forecast

plt.style.use('ggplot')


parser = argparse.ArgumentParser()
parser.add_argument('--best', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--test-size', default=7)
args = parser.parse_args()


class ARWisardEstimator(wp.RegressionWisard):
    def __init__(self, series, thermometer, *args, order=(0, 0, 0), **kwargs):
        super().__init__(*args, **kwargs)
        self.series = series
        self.p, self.d, self.q = order
        self.p += 1
        self.d += 1
        self.q += 1
        self.t_size, self.t_min, self.t_max = thermometer
        self._create_thermometer()
        self._moving_average()
        self._ts2supervised()

    def _create_thermometer(self):
        self.therm = wp.SimpleThermometer(self.t_size, self.t_min, self.t_max)

    def _moving_average(self):
        self.series = self.series.rolling(int(self.q)).mean()

    def _ts2supervised(self):
        supervised = self.series_to_supervised(self.series.values.tolist(), n_in=self.p)
        self.X_train = supervised.iloc[:, :-1].values
        self.y_train = supervised.iloc[:, -1].values

    def fit(self):
        train_ds = wp.DataSet()
        for i, x in enumerate(self.X_train):
            train_ds.add(self.therm.transform(x), self.y_train[i])
        super().train(train_ds)
        return self

    def forecast(self, steps=1):
        forecast = []
        x = self.X_train[-1]
        for _ in range(steps):
            test_ds = wp.DataSet()
            test_ds.add(self.therm.transform(x))
            y = super().predict(test_ds)[0]
            forecast.append(y)
            x = np.append(x[1:], [y])
        return np.array(forecast)

    @staticmethod
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        """
        Frame a time series as a supervised learning dataset.
        Arguments:
                data: Sequence of observations as a list or NumPy array.
                n_in: Number of lag observations as input (X).
                n_out: Number of observations as output (y).
                dropnan: Boolean whether or not to drop rows with NaN values.
        Returns:
                Pandas DataFrame of series framed for supervised learning.
        """
        n_vars = 1 if isinstance(data, list) else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg


def search_hyperparameters(train_ts: np.ndarray, test_ts: np.ndarray, criterion: str = 'mae'):
    p = q = range(0, 10)
    d = [0]
    t_size = np.arange(256, 1024, 256, dtype=int)
    t_min = np.linspace(train_ts.min(), train_ts.quantile(0.25), 5, dtype=float)
    t_max = np.linspace(train_ts.quantile(0.75), train_ts.max(), 5, dtype=float)
    addrs = np.arange(5, 25, dtype=int)
    pdq = list(itertools.product(p, d, q))
    t_sz_min_max = list(itertools.product(t_size, t_min, t_max))
    means = [wp.SimpleMean(), wp.Median(), wp.GeometricMean(), wp.PowerMean(2), wp.HarmonicMean(), wp.ExponentialMean()]
    pmdf = pd.DataFrame()
    for mean in tqdm(means, dynamic_ncols=True):
        for order in tqdm(pdq, leave=False, dynamic_ncols=True):
            for thermometer in tqdm(t_sz_min_max, leave=False, dynamic_ncols=True):
                for addr in tqdm(addrs, leave=False, dynamic_ncols=True):
                    mod = ARWisardEstimator(train_ts, thermometer, addr, order=order, mean=mean)
                    results = mod.fit()
                    forecast = results.forecast(steps=args.test_size)
                    _rmse = rmse(test_ts.values, forecast)
                    _mape = mape(test_ts.values, forecast)
                    _mpe = mpe(test_ts.values, forecast)
                    _mae = mae(test_ts.values, forecast)
                    pmdf = pmdf.append({'mean': str(mean), 'thermometer': thermometer, 'addr': addr, 'order': order,
                                        'rmse': _rmse, 'mape': _mape, 'mae': _mae, 'mpe': _mpe}, ignore_index=True)
    pmdf['addr'] = pmdf['addr'].astype(int)
    best = pmdf[pmdf[criterion].abs().eq(pmdf[criterion].abs().min())].iloc[0]
    return pmdf, best


@profile(precision=4, stream=open(f"{os.path.join('results', 'arwisard')}/{args.dataset.split('/')[-2]}.log", 'w+'))
def fit_predict(train_ts: np.ndarray, best: pd.Series):
    # TODO Replace simple mean with the best one
    model = ARWisardEstimator(train_ts, best['thermometer'], best['addr'], order=best['order'], mean=wp.SimpleMean()).fit()
    forecast = model.forecast(steps=args.test_size)
    return model, forecast


def log_results(pmdf, forecast):
    if 'covid' in args.dataset:
        ds_name = 'covid'
    elif 'temperatures' in args.dataset:
        ds_name = 'temperatures'
    else:
        ds_name = 'synthetic'
    results_dir = os.path.join('results', 'arwisard')
    os.makedirs(results_dir, exist_ok=True)

    plot_observed_vs_forecast(
        os.path.join(results_dir, f'{ds_name}_inference.png'),
        test_ts.values,
        forecast,
        title=f'ARIMA - Inferência de {args.test_size} no dataset {args.dataset}')

    pmdf.to_csv(os.path.join(results_dir, f'{ds_name}_metrics.txt'), index=False)


if __name__ == '__main__':
    ts = pd.read_csv(args.dataset, index_col=0, squeeze=True)

    if isinstance(args.test_size, float):
        args.test_size = (ts.size * args.test_size)
    train_ts = ts.iloc[:-args.test_size]
    test_ts = ts.iloc[-args.test_size:]

    criterion = 'mae'
    if not args.best:
        pmdf, best = search_hyperparameters(train_ts, test_ts, criterion)
    else:
        pmdf = pd.read_csv(args.best, index_col=0)
        pmdf.columns = ['addr', 'mae', 'mape', 'mpe', 'order', 'rmse', 'thermometer']
        pmdf['order'] = pmdf['order'].apply(lambda x: tuple(map(int, x.strip('(').strip(')').split(','))))
        pmdf['thermometer'] = pmdf['thermometer'].apply(lambda x: tuple(map(int, map(float, x.strip('(').strip(')').split(',')))))
        best = pmdf[pmdf[criterion].eq(pmdf[criterion].min())].iloc[0]

    start = time.time()
    model, forecast = fit_predict(train_ts, best)
    print(f'Elapsed time: {round(time.time() - start, 2)} seconds')

    log_results(pmdf, forecast)
