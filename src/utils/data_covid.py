import pandas as pd
import requests
import json

from utils.plot import plot_time_series


if __name__ == "__main__":
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:82.0) Gecko/20100101 Firefox/82.0',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.5',
        'Referer': 'https://dadoscovid19.cos.ufrj.br/pt',
        'Content-Type': 'application/json',
        'X-CSRFToken': 'undefined',
        'Origin': 'https://dadoscovid19.cos.ufrj.br',
        'Connection': 'keep-alive',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache',
    }

    data = '{"output":"my-graph1.figure","outputs":{"id":"my-graph1","property":"figure"},"inputs":[{"id":"filtro-idade","property":"value","value":["0001","0114","1524","2559","6060"]},{"id":"filtro-sexo","property":"value","value":[]},{"id":"filtro-uf","property":"value","value":["RJ"]},{"id":"filtro-local","property":"value","value":["RIO DE JANEIRO"]},{"id":"url","property":"pathname","value":"/pt"}],"changedPropIds":["filtro-sexo.value"]}'

    response = json.loads(
        requests.post(
            'https://dadoscovid19.cos.ufrj.br/_dash-update-component',
            headers=headers,
            data=data).text)

    x = pd.to_datetime(response['response']['my-graph1']['figure']['data'][1]['x'])
    y = response['response']['my-graph1']['figure']['data'][1]['y']
    series = pd.Series(y, index=x)

    plot_time_series(
        "data/covid/img/confirmed_cases.png",
        series.index,
        series.values,
        "Number of Confirmed Cases of Covid-19")

    series.to_csv("data/covid/confirmed_cases.csv")
