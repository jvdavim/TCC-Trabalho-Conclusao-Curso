import os

import pandas as pd

if __name__ == '__main__':
    DATA_URL = 'http://sistemas.saude.rj.gov.br/tabnetbd/csv/esus_sivep16255351829.csv'
    DATA_DIR = os.path.join('data', 'covid')

    df = pd.read_csv(DATA_URL, sep=';', names=['data', 'casos_confirmados', 'obitos_confirmados'],
                     skiprows=2).iloc[:-1, :]
    df['data'] = pd.to_datetime(df['data'])

    os.makedirs(DATA_DIR, exist_ok=True)

    df[['data', 'casos_confirmados']].to_csv(os.path.join(DATA_DIR, 'casos_confirmados.csv'),
                                             index=False,
                                             sep=';',
                                             date_format='%Y-%m-%d')

    df[['data', 'obitos_confirmados']].to_csv(os.path.join(DATA_DIR, 'obitos_confirmados.csv'),
                                              index=False,
                                              sep=';',
                                              date_format='%Y-%m-%d')
