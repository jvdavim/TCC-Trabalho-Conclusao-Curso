import datetime
import os
import warnings
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fbprophet import Prophet
from memory_profiler import profile

from utils.plot import plot_observed_vs_forecast

warnings.filterwarnings('ignore')
plt.style.use('ggplot')

parser = argparse.ArgumentParser()
parser.add_argument('--config-set-path', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--dataset-path', type=str)
parser.add_argument('--opt-metric', type=str, default='mae')
parser.add_argument('--test-size', default=7)
args = vars(parser.parse_args())

DATASET = args['dataset']
DATASET_PATH = args['dataset_path']
CONFIG_SET_PATH = args['config_set_path']
OPT_METRIC = args['opt_metric']
TEST_SIZE = args['test_size']

if DATASET == 'covid':
    CAP = 2000
    FLOOR = 0
elif DATASET == 'temperatures':
    CAP = 30
    FLOOR = -5
else:
    CAP = None
    FLOOR = None


@profile(precision=4, stream=open(f"{os.path.join('results', 'prophet')}/{DATASET}.log", 'w+'))
def fit_predict(test_ts: np.ndarray, best: dict):
    m = Prophet(**best).fit(test_ts)  # Fit model with given params
    future = m.make_future_dataframe(periods=TEST_SIZE)
    future['floor'] = FLOOR
    future['cap'] = CAP
    forecast = m.predict(future)
    p = forecast.loc[:, 'yhat'].iloc[-TEST_SIZE:].values
    return m, p


def main():
    # Load configs set with metrics and hyperparameters
    config_set_df = pd.read_csv(CONFIG_SET_PATH,
                                dtype={
                                    "growth": str,
                                    "n_changepoints": int,
                                    "changepoint_range": float,
                                    "yearly_seasonality": bool,
                                    "weekly_seasonality": bool,
                                    "daily_seasonality": bool,
                                    "seasonality_mode": str,
                                    "seasonality_prior_scale": float,
                                    "holidays_prior_scale": float,
                                    "changepoint_prior_scale": float,
                                    "mcmc_samples": int,
                                    "interval_width": float,
                                    "uncertainty_samples": int
                                })

    # Load best hyperparameters
    best_params = config_set_df[config_set_df[OPT_METRIC].abs().eq(config_set_df[OPT_METRIC].abs().min())].iloc[0:1, :]
    best_params = {k: v for k, v in best_params.iloc[0].iteritems()}
    best_params['changepoints'] = None
    best_params['holidays'] = None
    best_params['stan_backend'] = None
    best_params = {k: v for k, v in best_params.items() if k in Prophet().__dict__.keys()}

    # Load test dataset
    ts = pd.read_csv(DATASET_PATH)
    ts.columns = ['ds', 'y']
    if 'synthetic' in DATASET_PATH:
        ts['ds'] = pd.date_range(start=datetime.date(2020, 12, 14) - datetime.timedelta(days=len(ts.ds)),
                                 end=datetime.date(2020, 12, 13))
    test_ts = ts.iloc[-TEST_SIZE:]
    test_ts['floor'] = FLOOR
    test_ts['cap'] = CAP

    model, pred = fit_predict(test_ts, best_params)

    results_dir = os.path.join('results', 'prophet')
    os.makedirs(results_dir, exist_ok=True)

    plot_observed_vs_forecast(os.path.join(results_dir, f'{DATASET}_inference.png'),
                              test_ts.values,
                              pred,
                              title=f"Prophet - Inferência de {TEST_SIZE} no dataset {DATASET}")

    model.plot_components(pred)
    plt.savefig(os.path.join(results_dir, f'{DATASET}_diagnostics.png'))


if __name__ == '__main__':
    main()
