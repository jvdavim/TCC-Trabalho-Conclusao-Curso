"""
This script is responsable for training and testing ARIMA models. 
Input: 
Given by the argparse lib. 
Output:
- results/arima/<DATASET_NAME>_grid_results.csv
"""

import argparse
import csv
import itertools
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
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

if __name__ == '__main__':
    DATSET_NAME = args.dataset.split('/')[-1].split('.')[0]
    RESULTS_DIR = os.path.join('results', 'arima')

    # Load dataset
    ts = pd.read_csv(args.dataset, index_col=0, sep=';')

    # Split train and test
    if isinstance(args.test_size, float):
        args.test_size = (ts.size * args.test_size)
    train_ts = ts.iloc[:-args.test_size - args.val_size].values
    val_ts = ts.iloc[-args.test_size - args.val_size:-args.test_size].values
    test_ts = ts.iloc[-args.test_size:].values

    # Grid parameters
    p = d = q = range(0, 10)
    pdq = list(itertools.product(p, d, q))

    # Search hyperparameters
    if args.search_hyperparameters:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f'{DATSET_NAME}_grid_results.csv'), 'w') as output_file:
            header = ['p', 'q', 'd', 'aic', 'rmse', 'mape', 'mae', 'mpe']
            writer = csv.DictWriter(output_file, fieldnames=header, delimiter=';')
            writer.writeheader()
            for param in tqdm(pdq, dynamic_ncols=True):
                model = ARIMA(train_ts, order=param, enforce_stationarity=False, enforce_invertibility=False)
                results = model.fit()
                forecast = results.forecast(steps=args.test_size)
                writer.writerow({
                    'p': param[0],
                    'd': param[1],
                    'q': param[2],
                    'aic': results.aic,
                    'rmse': rmse(val_ts, forecast),
                    'mape': mape(val_ts, forecast),
                    'mae': mae(val_ts, forecast),
                    'mpe': mpe(val_ts, forecast)
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
            param = (best.p, best.d, best.q)

            # Train ARIMA model with selected hyperparams and train + validation data
            model = ARIMA(np.concatenate((train_ts, val_ts)),
                          order=param,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            results = model.fit()
            forecast = results.forecast(steps=args.test_size)

            writer.writerow({
                'opt_metric': opt_metric,
                'rmse': rmse(test_ts, forecast),
                'mape': mape(test_ts, forecast),
                'mae': mae(test_ts, forecast),
                'mpe': mpe(test_ts, forecast)
            })
            output_file.flush()
