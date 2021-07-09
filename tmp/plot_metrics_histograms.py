import os

import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')


best = lambda df, col: df[df[col].abs().eq(df[col].abs().min())].loc[:, col].iloc[0]


if __name__ == '__main__':
    arima_temperatures = pd.read_csv('results/arima/temperatures_metrics.txt')
    arima_covid = pd.read_csv('results/arima/covid_metrics.txt')
    arima_synthetic = pd.read_csv('results/arima/synthetic_metrics.txt')
    arwisard_temperatures = pd.read_csv('results/arwisard/temperatures_metrics.txt')
    arwisard_covid = pd.read_csv('results/arwisard/covid_metrics.txt')
    arwisard_synthetic = pd.read_csv('results/arwisard/synthetic_metrics.txt')
    prophet_temperatures = pd.read_csv('results/prophet/temperatures_metrics.txt')
    prophet_covid = pd.read_csv('results/prophet/covid_metrics.txt')
    prophet_synthetic = pd.read_csv('results/prophet/synthetic_metrics.txt')

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

    os.makedirs('results', exist_ok=True)
    datasets = df['dataset'].unique()

    for dataset in datasets:
        df[df['dataset'] == dataset].pivot('dataset', 'model', 'rmse').plot.bar(title=f'{dataset} rmse')
        plt.savefig(f'results/metrics/{dataset}_rmse_histogram.png')

        df[df['dataset'] == dataset].pivot('dataset', 'model', 'mape').plot.bar(title=f'{dataset} mape')
        plt.savefig(f'results/metrics/{dataset}_mape_histogram.png')

        df[df['dataset'] == dataset].pivot('dataset', 'model', 'mpe').plot.bar(title=f'{dataset} mpe')
        plt.savefig(f'results/metrics/{dataset}_mpe_histogram.png')

        df[df['dataset'] == dataset].pivot('dataset', 'model', 'mae').plot.bar(title=f'{dataset} mae')
        plt.savefig(f'results/metrics/{dataset}_mae_histogram.png')
