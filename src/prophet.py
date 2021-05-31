import argparse
import datetime
import itertools
import os
import time
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from fbprophet import Prophet
from fbprophet.diagnostics import cross_validation, performance_metrics
from memory_profiler import profile
from tqdm import tqdm

from utils.metrics import *
from utils.plot import plot_observed_vs_forecast

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--best', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--test-size', default=7)
args = parser.parse_args()


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


def search_hyperparameters(train_ts: np.ndarray, criterion: str = 'mae'):
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
        'stan_backend': None,
        'cap': None,
        'floor': None
    }
    if 'covid' in args.dataset:
        parameters['growth'] = 'logistic'
        parameters['cap'] = 2000
        parameters['floor'] = 0
        parameters['daily_seasonality'] = False
        parameters['weekly_seasonality'] = True
        parameters['yearly_seasonality'] = False

    # TODO Add other dataset parameters

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
    for params in tqdm(all_params):
        with suppress_stdout_stderr():
            m = Prophet(**params).fit(train_ts)  # Fit model with given params
            df_cv = cross_validation(m, horizon=f'{args.test_size} days', parallel='processes')
            df_p = performance_metrics(df_cv, rolling_window=1)
            pmdf = pmdf.append({**params, **dict(zip(df_p.columns, df_p.iloc[0, :].values))}, ignore_index=True)

    best = pmdf[pmdf[criterion].abs().eq(pmdf[criterion].abs().min())].iloc[0]

    return pmdf, best


@profile(precision=4, stream=open(f"{os.path.join('results', 'prophet')}/{args.dataset.split('/')[-2]}.log", 'w+'))
def fit_predict(train_ts: np.ndarray, best: pd.Series):
    model = Prophet(**best.to_dict())
    model.fit(train_ts)
    future = model.make_future_dataframe(periods=args.test_size)
    forecast = model.predict(future)
    return model, forecast


def log_results(pmdf, forecast, model):
    if 'covid' in args.dataset:
        ds_name = 'covid'
    elif 'temperatures' in args.dataset:
        ds_name = 'temperatures'
    else:
        ds_name = 'synthetic'
    results_dir = os.path.join('results', 'arima')
    os.makedirs(results_dir, exist_ok=True)

    try:
        plot_observed_vs_forecast(
            os.path.join(results_dir, f'{ds_name}_inference.png'),
            test_ts.values,
            forecast.iloc[-args.test_size:, :].values,
            title=f'Prophet - Inferência de {args.test_size} no dataset {ds_name}')
    except Exception as ex:
        print(f'Erro: {ex}')

    try:
        fig2 = model.plot_components(forecast)
        plt.savefig(os.path.join(results_dir, f'{ds_name}_diagnostics.png'))
    except Exception as ex:
        print(f'Erro: {ex}')

    pmdf.to_csv(os.path.join(results_dir, f'{ds_name}_metrics.txt'))


if __name__ == '__main__':
    ts = pd.read_csv(args.dataset)
    ts.columns = ['ds', 'y']
    if 'synthetic' in args.dataset:
        ts['ds'] = pd.date_range(start=datetime.date(2020, 12, 14) - datetime.timedelta(days=len(ts.ds)),
                                 end=datetime.date(2020, 12, 13))

    # Split train and test data
    if isinstance(args.test_size, float):
        args.test_size = int(len(ts) * args.test_size)
    train_ts = ts.iloc[:-args.test_size, :]
    test_ts = ts.iloc[-args.test_size:]

    pmdf, best = search_hyperparameters(train_ts, criterion='mae')

    start = time.time()
    try:
        model, forecast = fit_predict(train_ts, best)
    except Exception as ex:
        print(f'Erro: {ex}')
    print(f'Elapsed time: {round(time.time() - start, 2)} seconds')

    log_results(pmdf, forecast, model)
