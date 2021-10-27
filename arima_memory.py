import argparse
import os

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, help='')
parser.add_argument('--dataset-name', type=str, help='')
parser.add_argument('--metric', type=str, help='')
parser.add_argument('--test-size', type=int, help='')
args = parser.parse_args()


@profile
def profile_memory(ts, order, test_size) -> None:
    model = ARIMA(ts, order=order, enforce_stationarity=False, enforce_invertibility=False).fit()
    _ = model.forecast(steps=test_size)


def main() -> None:
    ts = pd.read_csv(args.dataset_path, index_col=0, sep=';').values
    grid_results_df = pd.read_csv(os.path.join('results/arima', f"{args.dataset_name}_grid_results.csv"), sep=';')
    best = grid_results_df[grid_results_df[args.metric].abs().eq(grid_results_df[args.metric].abs().min())].iloc[0]
    order = (best.p, best.d, best.q)
    profile_memory(ts, order, args.test_size)


if __name__ == '__main__':
    main()
