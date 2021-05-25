import os

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')


def best(df, col):
    return df[df[col].abs().eq(df[col].abs().min())].loc[:, col].iloc[0]


if __name__ == '__main__':
    arima_temperatures = pd.read_csv("results/multistep/arima/temperatures_metrics.txt", index_col=0)
    arima_covid = pd.read_csv("results/multistep/arima/covid_metrics.txt", index_col=0)
    arima_synthetic = pd.read_csv("results/multistep/arima/synthetic_metrics.txt", index_col=0)
    arwisard_temperatures = pd.read_csv("results/multistep/arwisard/temperatures_metrics.txt", index_col=0)
    arwisard_covid = pd.read_csv("results/multistep/arwisard/covid_metrics.txt", index_col=0)
    arwisard_synthetic = pd.read_csv("results/multistep/arwisard/synthetic_metrics.txt", index_col=0)
    prophet_temperatures = pd.read_csv("results/multistep/prophet/temperatures_metrics.txt", index_col=0)
    prophet_covid = pd.read_csv("results/multistep/prophet/covid_metrics.txt", index_col=0)
    prophet_synthetic = pd.read_csv("results/multistep/prophet/synthetic_metrics.txt", index_col=0)

    df = pd.DataFrame([
        ['arima', 'temperatures', best(arima_temperatures, 'rmse'), best(arima_temperatures, 'mape'), best(arima_temperatures, 'mpe'), best(arima_temperatures, 'mae')],
        ['arima', 'covid', best(arima_covid, 'rmse'), best(arima_covid, 'mape'), best(arima_covid, 'mpe'), best(arima_covid, 'mae')],
        ['arima', 'synthetic', best(arima_synthetic, 'rmse'), best(arima_synthetic, 'mape'), best(arima_synthetic, 'mpe'), best(arima_synthetic, 'mae')],
        ['arwisard', 'temperatures', best(arwisard_temperatures, 'rmse'), best(arwisard_temperatures, 'mape'), best(arwisard_temperatures, 'mpe'), best(arwisard_temperatures, 'mae')],
        ['arwisard', 'covid', best(arwisard_covid, 'rmse'), best(arwisard_covid, 'mape'), best(arwisard_covid, 'mpe'), best(arwisard_covid, 'mae')],
        ['arwisard', 'synthetic', best(arwisard_synthetic, 'rmse'), best(arwisard_synthetic, 'mape'), best(arwisard_synthetic, 'mpe'), best(arwisard_synthetic, 'mae')],
        ['prophet', 'temperatures', best(prophet_temperatures, 'rmse'), best(prophet_temperatures, 'mape'), best(prophet_temperatures, 'mpe'), best(prophet_temperatures, 'mae')],
        ['prophet', 'covid', best(prophet_covid, 'rmse'), best(prophet_covid, 'mape'), best(prophet_covid, 'mpe'), best(prophet_covid, 'mae')],
        ['prophet', 'synthetic', best(prophet_synthetic, 'rmse'), best(prophet_synthetic, 'mape'), best(prophet_synthetic, 'mpe'), best(prophet_synthetic, 'mae')]
    ], columns=['model', 'dataset', 'rmse', 'mape', 'mpe', 'mae'])

    os.makedirs("results/multistep", exist_ok=True)
    datasets = df['dataset'].unique()

    for dataset in datasets:
        df[df['dataset'] == dataset].pivot("dataset", "model", "rmse").plot.bar(title=f"{dataset} rmse")
        plt.savefig(f"results/multistep/{dataset}_rmse_histogram.png")

        df[df['dataset'] == dataset].pivot("dataset", "model", "mape").plot.bar(title=f"{dataset} mape")
        plt.savefig(f"results/multistep/{dataset}_mape_histogram.png")

        df[df['dataset'] == dataset].pivot("dataset", "model", "mpe").plot.bar(title=f"{dataset} mpe")
        plt.savefig(f"results/multistep/{dataset}_mpe_histogram.png")

        df[df['dataset'] == dataset].pivot("dataset", "model", "mae").plot.bar(title=f"{dataset} mae")
        plt.savefig(f"results/multistep/{dataset}_mae_histogram.png")
