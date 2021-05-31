import argparse
import itertools
import os
import time
import warnings
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from memory_profiler import profile
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from tqdm import tqdm

from utils.metrics import mae, mape, mpe, rmse
from utils.plot import plot_observed_vs_forecast

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--best', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--test-size', default=7)
args = parser.parse_args()


def search_hyperparameters(train_ts: np.ndarray, test_ts: np.ndarray, criterion: str = 'aic') -> Tuple[pd.DataFrame, pd.Series]:
    """Search best hyperparameters for dataset.

    Args:
        train_ts (np.ndarray): Train time series.
        criterion (str, optional): Criterion to minimize the error. Can be any metric used
        in the experiment. Defaults to 'aic'.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Return a pandas dataframe with all runs results and a pandas series with the
         best one.
    """
    p = d = q = range(0, 10)
    pdq = list(itertools.product(p, d, q))
    pmdf = pd.DataFrame()
    for param in tqdm(pdq, desc='Searching best hyperparameters'):
        mod = ARIMA(train_ts, order=param, enforce_stationarity=False, enforce_invertibility=False)
        results = mod.fit()
        forecast = results.forecast(steps=args.test_size)
        mape_ = mape(test_ts.values, forecast)
        rmse_ = rmse(test_ts.values, forecast)
        mpe_ = mpe(test_ts.values, forecast)
        mae_ = mae(test_ts.values, forecast)
        pmdf = pmdf.append({'order': param, 'aic': results.aic, 'rmse': rmse_,
                            'mape': mape_, 'mae': mae_, 'mpe': mpe_}, ignore_index=True)
    best = pmdf[pmdf[criterion].abs().eq(pmdf[criterion].abs().min())].iloc[0]
    return pmdf, best


@profile(precision=4,
         stream=open(f"{os.path.join('results', 'arima')}/{args.dataset.split('/')[-2]}.log", 'w+'))
def fit_predict(train_ts: np.ndarray, best: pd.Series) -> Tuple[ARIMAResults, np.ndarray]:
    model = ARIMA(train_ts, order=best['order'], enforce_stationarity=False, enforce_invertibility=False).fit()
    forecast = model.forecast(steps=args.test_size)
    return model, forecast


def log_results(pmdf, model, forecast):
    if 'covid' in args.dataset:
        ds_name = 'covid'
    elif 'temperatures' in args.dataset:
        ds_name = 'temperatures'
    else:
        ds_name = 'synthetic'
    results_dir = os.path.join('results', 'arima')
    os.makedirs(results_dir, exist_ok=True)

    model.plot_diagnostics(figsize=(20, 14))
    plt.savefig(os.path.join(results_dir, f'{ds_name}_diagnostics.png'))

    plot_observed_vs_forecast(
        os.path.join(results_dir, f'{ds_name}_inference.png'),
        test_ts.values,
        forecast,
        title=f'ARIMA - Inferência de {args.test_size} no dataset {args.dataset}')

    pmdf.to_csv(os.path.join(results_dir, f'{ds_name}_metrics.txt'))


if __name__ == '__main__':
    ts = pd.read_csv(args.dataset, index_col=0, squeeze=True)

    if isinstance(args.test_size, float):
        args.test_size = (ts.size * args.test_size)
    train_ts = ts.iloc[:-args.test_size]
    test_ts = ts.iloc[-args.test_size:]

    criterion = 'aic'
    if not args.best:
        pmdf, best = search_hyperparameters(train_ts, test_ts, criterion)
    else:
        pmdf = pd.read_csv(args.best, index_col=0)
        pmdf.columns = ['aic', 'mae', 'mape', 'mpe', 'order', 'rmse']
        pmdf['order'] = pmdf['order'].apply(lambda x: tuple(map(int, x.strip('(').strip(')').split(','))))
        best = pmdf[pmdf[criterion].eq(pmdf[criterion].min())].iloc[0]

    start = time.time()
    model, forecast = fit_predict(train_ts, best)
    print(f'Elapsed time: {round(time.time() - start, 2)} seconds')

    log_results(pmdf, model, forecast)
