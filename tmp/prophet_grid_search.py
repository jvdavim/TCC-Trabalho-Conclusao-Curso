import argparse
import datetime
import itertools
import os
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbprophet import Prophet
from tqdm import tqdm

from utils.metrics import *

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--best', type=str)
parser.add_argument('--criterion', type=str, default='mae')
parser.add_argument('--dataset', type=str)
parser.add_argument('--test-size', default=7)
parser.add_argument('--cap', type=int)
parser.add_argument('--floor', type=int)
args = vars(parser.parse_args())

parameters = {
    'growth': 'linear',
    'changepoints': None,
    'n_changepoints': 25,
    'changepoint_range': 0.8,
    'yearly_seasonality': 'auto',
    'weekly_seasonality': 'auto',
    'daily_seasonality': 'auto',
    'holidays': None,
    'seasonality_mode': 'additive',
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
    'changepoint_prior_scale': 0.05,
    'mcmc_samples': 0,
    'interval_width': 0.80,
    'uncertainty_samples': 1000,
    'stan_backend': None
}


class suppress_stdout_stderr(object):
    """A context manager for doing a 'deep suppression' of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).
    """

    def __init__(self):
        # Open a pair of null files
        self.null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0], 1)
        os.dup2(self.null_fds[1], 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0], 1)
        os.dup2(self.save_fds[1], 2)
        # Close the null files
        for fd in self.null_fds + self.save_fds:
            os.close(fd)


def prepare_logs() -> Tuple[str, os.PathLike]:
    if 'covid' in args['dataset']:
        ds_name = 'covid'
    elif 'temperatures' in args['dataset']:
        ds_name = 'temperatures'
    else:
        ds_name = 'synthetic'
    results_dir = os.path.join('results', 'prophet')
    os.makedirs(results_dir, exist_ok=True)
    return ds_name, results_dir


def prepare_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.read_csv(args['dataset'])
    ts.columns = ['ds', 'y']
    if 'synthetic' in args['dataset']:
        ts['ds'] = pd.date_range(start=datetime.date(2020, 12, 14) - datetime.timedelta(days=len(ts.ds)),
                                 end=datetime.date(2020, 12, 13))

    if isinstance(args['test_size'], float):
        args['test_size'] = int(len(ts) * args['test_size'])
    train_ts = ts.iloc[:-args['test_size'], :]
    test_ts = ts.iloc[-args['test_size']:]
    return train_ts, test_ts


def search_hyperparameters(train_ts: np.ndarray, criterion: str = 'mae'):
    if 'covid' in args['dataset']:
        parameters['growth'] = 'logistic'
        args['cap'] = 2000
        args['floor'] = 0
        parameters['daily_seasonality'] = False
        parameters['weekly_seasonality'] = True
        parameters['yearly_seasonality'] = False
    elif 'temperatures' in args['dataset']:
        parameters['growth'] = 'logistic'
        args['cap'] = 30
        args['floor'] = -5
        parameters['daily_seasonality'] = True
        parameters['weekly_seasonality'] = False
        parameters['yearly_seasonality'] = True
    else:
        parameters['growth'] = 'linear'
        parameters['daily_seasonality'] = False
        parameters['weekly_seasonality'] = True
        parameters['yearly_seasonality'] = False

    if parameters['growth'] == 'logistic':
        train_ts['floor'] = args['floor']
        train_ts['cap'] = args['cap']

    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'n_changepoints': [5, 15, 25],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    pmdf = pd.DataFrame()

    # Use cross validation to evaluate all parameters
    for params in tqdm(all_params, dynamic_ncols=True):
        with suppress_stdout_stderr():
            parameters.update(params)
            try:
                m = Prophet(**parameters).fit(train_ts)  # Fit model with given params
                future = m.make_future_dataframe(periods=args['test_size'])
                future['floor'] = args['floor']
                future['cap'] = args['cap']
                forecast = m.predict(future)
                predicted = forecast.loc[:, 'yhat'].iloc[-args['test_size']:].values
                _rmse = rmse(test_ts.y.values, predicted)
                _mape = mape(test_ts.y.values, predicted)
                _mpe = mpe(test_ts.y.values, predicted)
                _mae = mae(test_ts.y.values, predicted)
                pmdf = pmdf.append({
                    **parameters, 'rmse': _rmse,
                    'mape': _mape,
                    'mae': _mae,
                    'mpe': _mpe
                },
                                   ignore_index=True)
            except RuntimeError:
                print(f'[ERROR] Error with following hyperparameters: {parameters}')

    pmdf.to_csv(os.path.join(results_dir, f'{ds_name}_metrics.txt'), index=False)
    best = pmdf[pmdf[criterion].abs().eq(pmdf[criterion].abs().min())].iloc[0]

    return best


if __name__ == '__main__':
    ds_name, results_dir = prepare_logs()
    train_ts, test_ts = prepare_data()

    best = search_hyperparameters(train_ts, criterion=args['criterion'])
