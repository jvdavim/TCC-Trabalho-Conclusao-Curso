"""
This script is responsable for training and testing ARIMA models. 
Input: 
Given by the argparse lib. 
Output:
- results/arima/<DATASET_NAME>_grid_results.csv
"""

import argparse
import csv
import importlib
import itertools
import os
import re
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wisardpkg as wp
from tqdm import tqdm

from metrics import mae, mape, mpe, rmse

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='Path to data file.csv.')
parser.add_argument('--val-size', default=7, help='Validation size.')
parser.add_argument('--test-size', default=7, help='Test size')
parser.add_argument('--search-hyperparameters', action='store_true', help='Search hyperparameters.')
args = parser.parse_args()


class RegressionWisardEstimator(wp.RegressionWisard):

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


def get_name(self):
    m = re.search('\<wisardpkg\.(\w+).*', self.__str__())
    return m.group(1)


wp.SimpleMean.get_name = get_name
wp.Median.get_name = get_name
wp.GeometricMean.get_name = get_name
wp.PowerMean.get_name = get_name
wp.HarmonicMean.get_name = get_name
wp.ExponentialMean.get_name = get_name
# @profile(precision=4, stream=open(f"{os.path.join('results', 'arwisard')}/{args.dataset.split('/')[-2]}.log", 'w+'))
# def fit_predict(train_ts: np.ndarray, best: pd.Series):
#     # TODO Replace simple mean with the best one
#     model = ARWisardEstimator(train_ts, best['thermometer'], best['addr'], order=best['order'],
#                               mean=wp.SimpleMean()).fit()
#     forecast = model.forecast(steps=args.test_size)
#     return model, forecast

# def log_results(pmdf, forecast):
#     if 'covid' in args.dataset:
#         ds_name = 'covid'
#     elif 'temperatures' in args.dataset:
#         ds_name = 'temperatures'
#     else:
#         ds_name = 'synthetic'
#     results_dir = os.path.join('results', 'arwisard')
#     os.makedirs(results_dir, exist_ok=True)

#     plot_observed_vs_forecast(os.path.join(results_dir, f'{ds_name}_inference.png'),
#                               test_ts.values,
#                               forecast,
#                               title=f'ARIMA - Inferência de {args.test_size} no dataset {args.dataset}')

#     pmdf.to_csv(os.path.join(results_dir, f'{ds_name}_metrics.txt'), index=False)

if __name__ == '__main__':
    DATSET_NAME = args.dataset.split('/')[-1].split('.')[0]
    RESULTS_DIR = os.path.join('results', 'rew')

    # Load dataset
    ts = pd.read_csv(args.dataset, index_col=0, sep=';')

    # Split train and test
    if isinstance(args.test_size, float):
        args.test_size = (ts.size * args.test_size)
    train_ts = ts.iloc[:-args.test_size - args.val_size]
    val_ts = ts.iloc[-args.test_size - args.val_size:-args.test_size]
    test_ts = ts.iloc[-args.test_size:]

    # Grid parameters
    p = q = range(0, 10)
    d = [0]
    t_size = np.arange(256, 1024, 256, dtype=int)
    t_min = np.linspace(train_ts.min(), train_ts.quantile(0.25), 5, dtype=float, axis=-1)[0]
    t_max = np.linspace(train_ts.quantile(0.75), train_ts.max(), 5, dtype=float, axis=-1)[0]
    addrs = np.arange(5, 25, dtype=int)
    pdq = list(itertools.product(p, d, q))
    t_sz_min_max = list(itertools.product(t_size, t_min, t_max))
    means = [wp.SimpleMean(), wp.Median(), wp.GeometricMean(), wp.PowerMean(2), wp.HarmonicMean(), wp.ExponentialMean()]

    # Grid search
    # Search hyperparameters
    if args.search_hyperparameters:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f'{DATSET_NAME}_grid_results.csv'), 'w') as output_file:
            header = ['t_size', 't_min', 't_max', 'addr', 'mean_type', 'p', 'q', 'd', 'rmse', 'mape', 'mae', 'mpe']
            writer = csv.DictWriter(output_file, fieldnames=header, delimiter=';')
            writer.writeheader()
            for mean in tqdm(means, dynamic_ncols=True):
                for order in tqdm(pdq, leave=False, dynamic_ncols=True):
                    for thermometer in tqdm(t_sz_min_max, leave=False, dynamic_ncols=True):
                        for addr in tqdm(addrs, leave=False, dynamic_ncols=True):
                            mod = RegressionWisardEstimator(train_ts, thermometer, addr, order=order, mean=mean)
                            results = mod.fit()
                            forecast = results.forecast(steps=args.test_size)
                            _rmse = rmse(test_ts.values, forecast)
                            _mape = mape(test_ts.values, forecast)
                            _mpe = mpe(test_ts.values, forecast)
                            _mae = mae(test_ts.values, forecast)
                            writer.writerow({
                                't_size': thermometer[0],
                                't_min': thermometer[1],
                                't_max': thermometer[2],
                                'addr': addr,
                                'mean_type': mean.get_name(),
                                'p': order[0],
                                'd': order[1],
                                'q': order[2],
                                'rmse': _rmse,
                                'mape': _mape,
                                'mae': _mae,
                                'mpe': _mpe
                            })
                            output_file.flush()

    grid_results_df = pd.read_csv(os.path.join(RESULTS_DIR, f'{DATSET_NAME}_grid_results.csv'), sep=';')

    # Test
    opt_metrics = ['rmse', 'mape', 'mae', 'mpe']
    with open(os.path.join(RESULTS_DIR, f'{DATSET_NAME}_test_results.csv'), 'w') as output_file:
        header = ['opt_metric', 'rmse', 'mape', 'mae', 'mpe']
        writer = csv.DictWriter(output_file, fieldnames=header, delimiter=';')
        writer.writeheader()

        for opt_metric in opt_metrics:
            # Select best hyperparameters set for the current opt_metric
            best = grid_results_df[grid_results_df[opt_metric].abs().eq(
                grid_results_df[opt_metric].abs().min())].iloc[0]
            thermometer = (best.t_size, best.t_min, best.t_max)
            order = (best.p, best.d, best.q)
            addr = best.addr
            module = importlib.import_module('wisardpkg')
            class_ = getattr(module, best.mean_type)
            mean = class_() if best.mean_type != 'PowerMean' else class_(2)

            # Train ReW model with selected hyperparams and train + validation data
            model = RegressionWisardEstimator(pd.concat([train_ts, val_ts]),
                                              thermometer,
                                              addr,
                                              order=order,
                                              mean=mean)
            results = model.fit()
            forecast = results.forecast(steps=args.test_size)

            writer.writerow({
                'opt_metric': opt_metric,
                'rmse': rmse(test_ts.values, forecast),
                'mape': mape(test_ts.values, forecast),
                'mae': mae(test_ts.values, forecast),
                'mpe': mpe(test_ts.values, forecast)
            })
            output_file.flush()
