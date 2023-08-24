# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import proplot as pplt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import math

# %%


def valid_metric(y_true, y_pred):
    r = pearsonr(y_true, y_pred)[0]

    MSE = mean_squared_error(y_true, y_pred)
    RMSE = math.sqrt(MSE)

    MAE = mean_absolute_error(y_true, y_pred)

    RB = np.sum(y_pred - y_true) / np.sum(y_true) * 100

    return r, RMSE, MAE, RB


# %%
def test_plot(ax, x, y, xlim, ylim, color='k', xlabel=None, ylabel=None, xerr=None, yerr=None, err=True):
    r, rmse, mae, rb = valid_metric(y_true=y, y_pred=x)

    if err == False:
        ax.scatter(x=x, y=y, s=2, marker='o', alpha=0.2, color=color)

    else:
        ax.errorbar(
            x=x,
            y=y,
            xerr=xerr,
            yerr=yerr,
            ecolor=color,
            capsize=0,
            ls="",
            c=color,
            marker="o",
            mew=1,
            mfc=color,
            ms=1,
            elinewidth=0.7,
            alpha=0.8,
        )

    ax.text(0.05, 0.65, f'$R$={r:.2f} \nRMSE={rmse:.2f}\nMAE={mae:.2f}\nRB={rb:.2f}', transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=ylabel)

    ax.plot(xlim, ylim, '--', linewidth=0.8, color='0.5', transform=ax.transData)


# %%
# basin
df_clas_yr = pd.read_csv(r'D:\ET\BasinCalc\wb_closure\class_mrb_et_mmyr.csv', index_col=0)
df_hget_yr = pd.read_csv(r'D:\ET\BasinCalc\wb_closure\hget_mrb_et_mmyr.csv', index_col=0)
df_flux_yr = pd.read_csv(r'D:\ET\BasinCalc\wb_closure\flux_mrb_et_mmyr.csv', index_col=0)
df_glma_yr = pd.read_csv(r'D:\ET\BasinCalc\wb_closure\glma_mrb_et_mmyr.csv', index_col=0)
# %%
df_clas_yrmn = df_clas_yr.agg(['mean', 'std'])
df_flux_yrmn = df_flux_yr.agg(['mean', 'std'])
df_glma_yrmn = df_glma_yr.agg(['mean', 'std'])
df_hget_yrmn = df_hget_yr.agg(['mean', 'std'])
# %%
# site
df_LE = pd.read_csv(r'D:\ET\MLResults\compare_list.csv', index_col=0)
df_LE_gl = df_LE.dropna(axis=0, how='any')
# %%
df_list = [df_hget_yrmn, df_flux_yrmn, df_glma_yrmn]
xlabels = ['HG-Land', 'FLUXCOM', 'GLEAM']
columns = ['hget (mm/month)', 'fluxet (mm/month)', 'Gleam36aet (mm/month)']

fig, axs = pplt.subplots(ncols=3, nrows=2, figsize=(8, 5), share=0, equal=True, dpi=400)
axs.format(abc='(a)', abcloc='l')

for ax, column in zip(axs[:3], columns):
    test_plot(
        ax=ax,
        x=df_LE_gl.loc[:, column],
        y=df_LE_gl.loc[:, 'EstimateLE (mm/month)'],
        ylabel='ET$_{FLUXNET}$ (mm/month)',
        err=False,
        color='#68b0ab',
        xlim=(-50, 300),
        ylim=(-50, 300),
    )
    ax.format(xlim=(-50, 300), ylim=(-50, 300), xlocator=100, ylocator=100)

for ax, df, xlabel in zip(axs[3:], df_list, xlabels):
    test_plot(
        ax=ax,
        x=df.iloc[0, :],
        y=df_clas_yrmn.iloc[0, :],
        xerr=df.iloc[1, :],
        yerr=df_clas_yrmn.iloc[1, :],
        ylabel='ET$_{CLASS}$ (mm/yr)',
        err=True,
        color='gray6',
        xlim=(0, 1500),
        ylim=(0, 1500),
    )
    ax.format(xlim=(0, 1500), ylim=(0, 1500), xlocator=500, ylocator=500)

axs[0].set(xlabel='ET$_{HG-Land}$ (mm/month)')
axs[1].set(xlabel='ET$_{FLUXCOM}$ (mm/month)')
axs[2].set(xlabel='ET$_{GLEAM}$ (mm/month)')

axs[3].set(xlabel='ET$_{HG-Land}$ (mm/yr)')
axs[4].set(xlabel='ET$_{FLUXCOM}$ (mm/yr)')
axs[5].set(xlabel='ET$_{GLEAM}$ (mm/yr)')

fig.savefig(f'D:\\ET\\Figures\\scatter_site_bsn.png', bbox_inches='tight', dpi=400)
# %%
