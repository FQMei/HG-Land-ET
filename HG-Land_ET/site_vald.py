# %%
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# %%
import math
import numpy as np
import pandas as pd
import xarray as xr
import calendar
import matplotlib.pyplot as plt
import proplot as pplt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gaussian_kde
# %%
params = {
    'font.family': 'sans-serif', 'font.sans-serif': 'Helvetica',
    'axes.titlesize': 10, 'axes.labelsize': 10,
    'tick.labelpad': 0.3, 'tick.len': 1.5,
    'tick.width': 0.3, 'tick.labelsize': 10,
    'tick.minor': True,
    'xtick.minor.size': 1.5, 'ytick.minor.size': 1.5,
    'legend.fontsize': 10, 'font.size': 9,
    'label.pad': 0.5, 'label.size': 10,
    'grid': False, 'gridminor': False,
    'grid.linestyle': '--', 'grid.linewidth': 0.5,
    'lines.linewidth': 1, 'meta.width': 1,
    'pdf.fonttype': 42,  
}
pplt.rc.update(params)
# %%


def test_plot(x, y, cmap, figsize, xlabel, ylabel):

    fig, ax = plt.subplots(figsize=figsize, dpi=400)

    r2 = r2_score(y, x)
    MSE = mean_squared_error(y, x)
    RMSE = math.sqrt(MSE)

    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)
    idx = z.argsort()
    x, y, z = x[idx],  y[idx], z[idx]

    ax.scatter(x=x, y=y,  c=z, s=2, marker='.', alpha=1, cmap=cmap)
    ax.plot(ax.get_ylim(), ax.get_ylim(), '--', linewidth=0.8,
            color='0.5', transform=ax.transData,)

    ax.text(0.05, 0.75, f'$R^2$ = {r2:.2f}\nRMSE = {RMSE:.2f}', transform=ax.transAxes)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    plt.axis('square')

    return fig, ax


# %%
cmap = LinearSegmentedColormap.from_list(
    'my_cmap', ['#d3d6db', 'indigo8', 'yellow6'], N=256)
cmap
# %%
# 01
test_path = r'D:\ET_FQM\ETsubject\MLResults\Deepforest_test.csv'
hget_site_fig_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\FigureA\hget_site.png'
# %%
df_test = pd.read_csv(test_path)
df_test['days'] = df_test.apply(lambda x: calendar.monthrange(
    x['Year'], x['Month'])[1], axis=1)
df_test
# %%
df_test['test (mm/mon)'] = df_test['test (W/m2)']*0.035*df_test['days']
df_test['predict (mm/mon)'] = df_test['predict (W/m2)']*0.035*df_test['days']
df_test
# %%
fig, ax = test_plot(df_test['predict (mm/mon)'], df_test['test (mm/mon)'], cmap, (2, 2),
                    'ET$_{HG-Land}$ (mm month$^{-1}$)', 'ET$_{FLUXNET}$ (mm month$^{-1}$)')
fig.savefig(hget_site_fig_path, bbox_inches='tight', dpi=400)
# %%
# 02
flux_path = r'D:\ET_FQM\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*.nc'
gleam_path = r'D:\ET_FQM\DataCollect\ETProducts\Gleam\v3.6a\monthly\E_1980-2021_GLEAM_v3.6a_MO.nc'
# %%
df_test = pd.read_csv(test_path)
flux = xr.open_mfdataset(flux_path)
gleama = xr.open_dataset(gleam_path)
# %%
for i in range(len(df_test)):

    year, month = df_test.iat[i, 4], df_test.iat[i, 5]
    lat, lon = df_test.iat[i, 6], df_test.iat[i, 7]
    index = df_test.index[i]
    time = f'{str(year)}-{str(month)}'

    # fluxcom
    days = flux.sel(time=time).time.dt.daysinmonth.data
    df_LE = flux.sel(lon=lon, lat=lat, method='nearest').LE.to_dataframe()
    et_flux = df_LE.at[time, 'LE']*days/2.45  # mm/mon
    df_test.loc[index, 'fluxet (mm/month)'] = et_flux

    # gleam3.6a
    et_gleam = gleama.sel(lon=lon, lat=lat, time=time, method='nearest').E.data
    df_test.loc[index, 'Gleam36aet (mm/month)'] = et_gleam

df_test.to_csv(r'D:\ET_FQM\ETsubject\MLResults\sites_compare.csv')
# %%
# 03
flux_site_fig_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\flux_site.png'
gleam_site_fig_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\gleam_site.png'
# %%
df_test = pd.read_csv(r'D:\ET_FQM\ETsubject\MLResults\sites_compare.csv')
df_test.reset_index(inplace=True)
# %%
# Fluxcom
fig, ax = test_plot(df_test['fluxet (mm/month)'], df_test['test'], cmap, (2, 2),
                    'ET$_{FLUXCOM}$ (mm month$^{-1})$', 'ET$_{FLUXNET}$ (mm month$^{-1})$')
fig.savefig(flux_site_fig_path, bbox_inches='tight', dpi=400)
# %%
# Gleam
df_gleam = df_test[['Gleam36aet (mm/month)', 'test']].copy().dropna(axis=0)
df_gleam.reset_index(drop=True, inplace=True)
df_gleam
# %%
fig, ax = test_plot(df_gleam['Gleam36aet (mm/month)'], df_gleam['test'], cmap, (2, 2),
                    'ET$_{GLEAM}$ (mm month$^{-1})$', 'ET$_{FLUXNET}$ (mm month$^{-1})$')
fig.savefig(gleam_site_fig_path, bbox_inches='tight', dpi=400)
