import argparse
import importlib
import os

import pandas as pd

from rew import RegressionWisardEstimator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-path', type=str, help='')
parser.add_argument('--dataset-name', type=str, help='')
parser.add_argument('--metric', type=str, help='')
parser.add_argument('--test-size', default=7, type=int, help='')
args = parser.parse_args()


@profile
def profile_memory(ts, thermometer, addr, order, mean, test_size) -> None:
    model = RegressionWisardEstimator(ts, thermometer, addr, order=order, mean=mean).fit()
    _ = model.forecast(steps=test_size)


def main() -> None:
    ts = pd.read_csv(args.dataset_path, index_col=0, sep=';')
    grid_results_df = pd.read_csv(os.path.join('results/rew', f"{args.dataset_name}_grid_results.csv"), sep=';')
    best = grid_results_df[grid_results_df[args.metric].abs().eq(grid_results_df[args.metric].abs().min())].iloc[0]
    thermometer = (best.t_size, best.t_min, best.t_max)
    order = (best.p, best.d, best.q)
    addr = best.addr
    module = importlib.import_module('wisardpkg')
    class_ = getattr(module, best.mean_type)
    mean = class_() if best.mean_type != 'PowerMean' else class_(2)
    profile_memory(ts, thermometer, addr, order, mean, args.test_size)


if __name__ == '__main__':
    main()
