import matplotlib.pyplot as plt


def plot_time_series(fname, time, values, title):
    plt.figure(figsize=(10, 6))
    plt.plot(time, values)
    plt.xlabel("Time", fontsize=20)
    plt.ylabel("Value", fontsize=20)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.savefig(fname)


def plot_observed_vs_forecast(fname, observed, forecast, title=None):
    _, ax = plt.subplots(figsize=(20, 5),)
    ax.plot(range(len(forecast)), forecast, color='r', label='previsto')
    ax.plot(range(len(forecast)), observed, color='b', label='observado')
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend()
    plt.savefig(fname)
