import argparse
import os

import numpy as np
import pandas as pd
from fbprophet import Prophet

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, help='')
parser.add_argument('--dataset-name', type=str, help='')
parser.add_argument('--metric', type=str, help='')
parser.add_argument('--test-size', type=int, help='')
args = parser.parse_args()


@profile
def profile_memory(ts, parameters, test_size) -> None:
    m = Prophet(**parameters).fit(ts)
    future = m.make_future_dataframe(periods=test_size)
    future['floor'] = ts.loc[:, 'floor'].iloc[0]
    future['cap'] = ts.loc[:, 'cap'].iloc[0]
    _ = m.predict(future)


def main() -> None:
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
    ts = pd.read_csv(args.dataset_path, index_col=0, sep=';').values
    grid_results_df = pd.read_csv(os.path.join('results/prophet', f"{args.dataset_name}_grid_results.csv"), sep=';')
    best = grid_results_df[grid_results_df[args.metric].abs().eq(grid_results_df[args.metric].abs().min())].iloc[0]
    best = grid_results_df[grid_results_df[args.metric].abs().eq(grid_results_df[args.metric].abs().min())].iloc[0]
    params = best.iloc[:-4].to_dict()
    params = {k: v for k, v in params.items() if not (np.isnan(v) if type(v) == np.float64 else False)}
    parameters.update(params)
    profile_memory(ts, parameters, args.test_size)


if __name__ == '__main__':
    main()
