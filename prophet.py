import argparse
import csv
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet
from tqdm import tqdm

from metrics import *

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Path to data file.csv.')
    parser.add_argument('--val-size', default=7, help='Validation size.')
    parser.add_argument('--test-size', default=7, help='Test size')
    parser.add_argument('--search-hyperparameters', action='store_true', help='Search hyperparameters.')
    args = parser.parse_args()

    DATSET_NAME = args.dataset.split('/')[-1].split('.')[0]
    RESULTS_DIR = os.path.join('results', 'prophet')

    # Grid parameters
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'n_changepoints': [5, 15, 25],
        'seasonality_mode': ['additive', 'multiplicative']
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    # Load dataset
    ts = pd.read_csv(args.dataset, sep=';')
    ts.columns = ['ds', 'y']

    if DATSET_NAME == 'casos_confirmados':
        parameters['growth'] = 'logistic'
        ts['floor'] = 0
        ts['cap'] = 2000
        parameters['daily_seasonality'] = False
        parameters['weekly_seasonality'] = True
        parameters['yearly_seasonality'] = False
    elif DATSET_NAME == 'temperaturas':
        parameters['growth'] = 'logistic'
        ts['cap'] = 30
        ts['floor'] = -5
        parameters['daily_seasonality'] = True
        parameters['weekly_seasonality'] = False
        parameters['yearly_seasonality'] = True
    else:
        ts['cap'] = None
        ts['floor'] = None
        parameters['daily_seasonality'] = False
        parameters['weekly_seasonality'] = True
        parameters['yearly_seasonality'] = False

    # Split train and test
    if isinstance(args.test_size, float):
        args.test_size = (ts.size * args.test_size)
    train_ts = ts.iloc[:-args.test_size - args.val_size]
    val_ts = ts.iloc[-args.test_size - args.val_size:-args.test_size]
    test_ts = ts.iloc[-args.test_size:]

    # Grid search
    if args.search_hyperparameters:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f'{DATSET_NAME}_grid_results.csv'), 'w') as output_file:
            header = [
                'growth', 'changepoints', 'n_changepoints', 'changepoint_range', 'yearly_seasonality',
                'weekly_seasonality', 'daily_seasonality', 'holidays', 'seasonality_mode', 'seasonality_prior_scale',
                'holidays_prior_scale', 'changepoint_prior_scale', 'mcmc_samples', 'interval_width', 'stan_backend',
                'uncertainty_samples', 'rmse', 'mape', 'mae', 'mpe'
            ]
            writer = csv.DictWriter(output_file, fieldnames=header, delimiter=';')
            writer.writeheader()
            for params in tqdm(all_params, dynamic_ncols=True):
                with suppress_stdout_stderr():
                    parameters.update(params)
                    try:
                        m = Prophet(**parameters).fit(train_ts)  # Fit model with given params
                        future = m.make_future_dataframe(periods=args.test_size)
                        future['floor'] = ts.loc[:, 'floor'].iloc[0]
                        future['cap'] = ts.loc[:, 'cap'].iloc[0]
                        forecast = m.predict(future)
                        predicted = forecast.loc[:, 'yhat'].iloc[-args.test_size:].values
                        _rmse = rmse(test_ts.y.values, predicted)
                        _mape = mape(test_ts.y.values, predicted)
                        _mpe = mpe(test_ts.y.values, predicted)
                        _mae = mae(test_ts.y.values, predicted)
                        writer.writerow({**parameters, 'rmse': _rmse, 'mape': _mape, 'mae': _mae, 'mpe': _mpe})
                    except RuntimeError:
                        print(f'[ERROR] Error with following hyperparameters: {parameters}')
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
            params = best.iloc[:-4].to_dict()
            params = {k: v for k, v in params.items() if not (np.isnan(v) if type(v) == np.float64 else False)}
            parameters.update(params)

            # Train ReW model with selected hyperparams and train + validation data
            try:
                m = Prophet(**parameters).fit(train_ts)  # Fit model with given params
                future = m.make_future_dataframe(periods=args.test_size)
                future['floor'] = ts.loc[:, 'floor'].iloc[0]
                future['cap'] = ts.loc[:, 'cap'].iloc[0]
                forecast = m.predict(future)
                predicted = forecast.loc[:, 'yhat'].iloc[-args.test_size:].values
                _rmse = rmse(test_ts.y.values, predicted)
                _mape = mape(test_ts.y.values, predicted)
                _mpe = mpe(test_ts.y.values, predicted)
                _mae = mae(test_ts.y.values, predicted)
                writer.writerow({'opt_metric': opt_metric, 'rmse': _rmse, 'mape': _mape, 'mae': _mae, 'mpe': _mpe})
            except RuntimeError:
                print(f'[ERROR] Error with following hyperparameters: {parameters}')
            output_file.flush()