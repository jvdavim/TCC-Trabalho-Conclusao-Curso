from utils.metrics import *
from utils.plot import plot_observed_vs_forecast

from fbprophet import Prophet

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import argparse
import datetime
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='Dataset name. Available choices are covid, temperatures or synthetic.')
    parser.add_argument(
        '--test-size',
        default=7,
        help='If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split. If int, represents the absolute number of test samples.')
    args = parser.parse_args()

    assert args.dataset in [
        'covid', 'temperatures', 'synthetic'], f"Dataset {args.dataset} doesn't exists. Available choices are covid, temperatures or synthetic."
    assert type(args.test_size) in [
        int, float], f"Type of --test-size can't be {type(args.test_size)}. Must be int or float instead."

    results_dir = os.path.join("results", "multistep", "prophet")
    os.makedirs(results_dir, exist_ok=True)

    filenames = [obj for obj in os.listdir(os.path.join("data", args.dataset)) if '.csv' in obj]
    if len(filenames) > 1:
        print("\nChoose one of these files to load as dataset:")
        print("    * " + "\n    * ".join(filenames))
        filename = input("Filename: ")
    elif len(filenames) == 1:
        filename = filenames[0]
    else:
        raise FileNotFoundError()
    filepath = os.path.join("data", args.dataset, filename)

    df = pd.read_csv(filepath)
    df.columns = ['ds', 'y']
    if args.dataset == 'synthetic':
        df['ds'] = pd.date_range(
            start=datetime.date(2020, 12, 14) - datetime.timedelta(days=len(df.ds)),
            end=datetime.date(2020, 12, 13))
    series = df.y

    # Split train and test data
    if isinstance(args.test_size, float):
        args.test_size = (series.size * args.test_size)
    train_data = df.iloc[:-args.test_size, :]
    test_data = series.iloc[-args.test_size:]

    pmdf = pd.DataFrame()
    m = Prophet()
    m.fit(train_data)
    future = m.make_future_dataframe(periods=args.test_size)
    forecast = m.predict(future)
    forecast7 = forecast.yhat[-1 * args.test_size:]
    _rmse = rmse(test_data.values, forecast7)
    _mape = mape(test_data.values, forecast7)
    _mpe = mpe(test_data.values, forecast7)
    _mae = mae(test_data.values, forecast7)
    pmdf = pmdf.append({'RMSE': _rmse, 'MAPE': _mape, 'MAE': _mae, 'MPE': _mpe}, ignore_index=True)

    print(f"RMSE: {rmse(test_data.values, forecast7)}")
    print(f"MAPE: {mape(test_data.values, forecast7)}")

    plot_observed_vs_forecast(
        os.path.join(results_dir, f"{args.dataset}_inference.png"),
        test_data.values,
        forecast7,
        title=f"Prophet - Inferência de {args.test_size} no dataset {args.dataset}")

    fig2 = m.plot_components(forecast)
    plt.savefig(os.path.join(results_dir, f"{args.dataset}_diagnostics.png"))

    pmdf.to_csv(os.path.join(results_dir, f"{args.dataset}_metrics.txt"))
