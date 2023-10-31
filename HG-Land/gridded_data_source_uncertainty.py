# %%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import math

# %%
df_train = pd.read_csv(r'D:\ET_FQM\ETsubject\DataProcess\TrainingList_Flux.csv')
# %%
df_sel = df_train[
    [
        'SiteID',
        'Year',
        'Month',
        'latitude',
        'longitude',
        'IGBP',
        'EstimateLE',
        'tmp',
        'pre',
        'vap',
        'pet',
        'wet',
        'frs',
        'rad',
        'wnd',
        'tsk',
        'prs',
        'fAPAR',
        'LAI',
    ]
]
# %%
air = xr.open_dataset(r'F:\NCEP DOE R2\air.2m.mon.mean_05deg.nc')
prate = xr.open_dataset(r'F:\NCEP DOE R2\prate.sfc.mon.mean_05deg.nc')
dswrf = xr.open_dataset(r'F:\NCEP DOE R2\dswrf.sfc.mon.mean_05deg.nc')
uswrf = xr.open_dataset(r'F:\NCEP DOE R2\uswrf.sfc.mon.mean_05deg.nc')
pres = xr.open_dataset(r'F:\NCEP DOE R2\pres.sfc.mon.mean_05deg.nc')
skt = xr.open_dataset(r'F:\NCEP DOE R2\skt.sfc.mon.mean_05deg.nc')
wspd = xr.open_dataset(r'F:\NCEP DOE R2\wspd.10m.mon.mean_05deg.nc')
fapar = xr.open_mfdataset(r'F:\vegetation\GIMMS_LAI-FPAR3g_v02\fAPAR_05deg\fAPAR_*.nc')
lai = xr.open_mfdataset(r'F:\vegetation\GIMMS_LAI-FPAR3g_v02\LAI_05deg\LAI_*.nc')
# %%
air_ = air.air - 273.15
prate_ = prate.prate * (prate.time.dt.days_in_month) * 24 * 3600
netrad = dswrf.dswrf - uswrf.uswrf


# %%
def add_var(df, key):
    year = df['Year']
    month = df['Month']
    time = str(year) + '-' + str(month)

    lat = df['latitude']
    lon = df['longitude']

    if key == 'lai':
        value = lai.lai.sel(time=time, lat=lat, lon=lon, method='nearest').compute().data
        return value
    elif key == 'fapar':
        value = fapar.fAPAR.sel(time=time, lat=lat, lon=lon, method='nearest').compute().data
        return value
    elif key == 'tmp':
        value = air_.sel(time=time, lat=lat, lon=lon, method='nearest').data
        return value
    elif key == 'precip':
        value = prate_.sel(time=time, lat=lat, lon=lon, method='nearest').data
        return value
    elif key == 'rad':
        value = netrad.sel(time=time, lat=lat, lon=lon, method='nearest').data
        return value
    elif key == 'prs':
        value = pres.pres.sel(time=time, lat=lat, lon=lon, method='nearest').data
        return value
    elif key == 'tsk':
        value = skt.skt.sel(time=time, lat=lat, lon=lon, method='nearest').data
        return value
    elif key == 'wnd':
        value = wspd.wspd.sel(time=time, lat=lat, lon=lon, method='nearest').data
        return value


# %%
col_name = [
    'lai_gimms',
    'fapar_gimms',
    'tmp_ncep',
    'precip_ncep',
    'rad_ncep',
    'prs_ncep',
    'tsk_ncep',
    'wnd_ncep',
]
keys = ['lai', 'fapar', 'tmp', 'precip', 'rad', 'prs', 'tsk', 'wnd']
for col, key in zip(col_name, keys):
    df_sel[col] = df_sel.apply(add_var, axis=1, args=(key,))
    print(key)

df_sel.to_csv(r'D:\ET\DataProcess\TrainingList_datasource_uncertainty.csv')
# %%
# train the model
# %%


def test_plot(x, y, color, xlabel, ylabel, xtext, ytext, label, ax, str=None):
    MSE = mean_squared_error(y, x)
    RMSE = math.sqrt(MSE)

    r = pearsonr(y, x)[0]

    MAE = mean_absolute_error(y, x)

    RB = np.sum(x - y) / np.sum(y) * 100

    ax.scatter(x=x, y=y, s=4, alpha=0.3, color=color, label=label)
    ax.plot(ax.get_ylim(), ax.get_ylim(), '--', linewidth=0.8, color='0.5', transform=ax.transData)

    ax.text(
        x=xtext,
        y=ytext,
        s=f'$R$ = {r:.2f}\nRMSE = {RMSE:.2f}\nMAE = {MAE:.2f}\nRB = {RB:.2f}',
        c=color,
        transform=ax.transAxes,
    )

    ax.text(0.01, 1.02, str, transform=ax.transAxes)

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_aspect('equal', adjustable='box')
    ax.legend()


# %%
df_uncert = pd.read_csv(r'D:\ET\MLResults\Deepf_test_results\Deepf_test_CRO_gridded.csv', index_col=0)
df_uncert['days'] = pd.PeriodIndex(year=df_uncert['Year'], month=df_uncert['Month'], freq='M').days_in_month
df_uncert['ET_true (mm/mon)'] = df_uncert['EstimateLE'] * 0.035 * df_uncert['days']
df_uncert['ET_pred (mm/mon)'] = df_uncert['predict (W/m2)'] * 0.035 * df_uncert['days']

df_cro = pd.read_csv(r'D:\ET\MLResults\Deepf_test_results\Deepf_test_CRO.csv', index_col=0)
df_cro['days'] = pd.PeriodIndex(year=df_cro['Year'], month=df_cro['Month'], freq='M').days_in_month
df_cro['ET_true (mm/mon)'] = df_cro['EstimateLE'] * 0.035 * df_cro['days']
df_cro['ET_pred (mm/mon)'] = df_cro['predict (W/m2)'] * 0.035 * df_cro['days']
# %%
fig, axs = plt.subplots(figsize=(4, 4), dpi=400)

test_plot(
    x=df_cro['ET_pred (mm/mon)'],
    y=df_cro['ET_true (mm/mon)'],
    color='gray',
    xlabel='ET$_{predict}$ (mm/month)$',
    ylabel='ET$_{FLUXNET}$ (mm/month)$',
    xtext=0.05,
    ytext=0.65,
    label='Source 1',
    ax=axs,
)

test_plot(
    x=df_uncert['ET_pred (mm/mon)'],
    y=df_uncert['ET_true (mm/mon)'],
    color='#F8A488',
    xlabel='ET$_{predict}$ (mm/month)',
    ylabel='ET$_{FLUXNET}$ (mm/month)',
    xtext=0.05,
    ytext=0.43,
    label='Source 2',
    ax=axs,
)

fig.savefig(f'D:\\ET\\Figures\\datasource_uncertainty_gridded.png', bbox_inches='tight', dpi=400)
