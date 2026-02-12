import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sb

def nan_replace(df):
    for col in df.columns:
        if df[col].isna().any():
            if is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                modul = df[col].mode()
                df[col].fillna(modul[0], inplace=True)

def to_dataframe(x, nume_randuri=None, nume_coloane=None, nume_fisier=None):
    df = pd.DataFrame(data=x, index=nume_randuri, columns=nume_coloane)
    if nume_fisier is not None:
        df.to_csv(nume_fisier)
    return df

# metode grafice
def corelograma(t, vmin=-1, vmax=1, titlu="Corelatii factoriale"):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax_ = sb.heatmap(t, vmin=vmin, vmax=vmax, cmap="RdYlBu", annot=True, ax=ax)
    ax_.set_xticklabels(t.columns, rotation=30, ha="right")
    plt.show()

def plot_corelatii(t, var_x, var_y, titlu="Plot corelatii", aspect="auto"):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(var_x, fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel(var_y, fontdict={"fontsize": 12, "color": "b"})
    ax.set_aspect(aspect)
    theta = np.arange(0, 2 * np.pi, 0.01)
    ax.plot(np.cos(theta), np.sin(theta), color="b")
    ax.axhline(0, color='gray', linestyle='--')
    ax.axvline(0, color='gray', linestyle='--')
    ax.scatter(t[var_x], t[var_y], color="r")
    for i in range(len(t)):
        ax.text(t[var_x].iloc[i], t[var_y].iloc[i], t.index[i])
    plt.show()

def plot_componente(t, var_x, var_y, titlu="Plot componente", aspect="auto"):
    fig = plt.figure(figsize=(13, 9))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(var_x, fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel(var_y, fontdict={"fontsize": 12, "color": "b"})
    ax.set_aspect(aspect)
    ax.scatter(t[var_x], t[var_y], color="r")
    for i in range(len(t)):
        ax.text(t[var_x].iloc[i], t[var_y].iloc[i], t.index[i], fontsize=9)
    plt.show()