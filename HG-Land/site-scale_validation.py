# %%
import math
import numpy as np
import pandas as pd
import xarray as xr
import calendar
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr


# %%
def valid_metric(y_true, y_pred):
    r = pearsonr(y_true, y_pred)[0]

    MSE = mean_squared_error(y_true, y_pred)
    RMSE = math.sqrt(MSE)

    MAE = mean_absolute_error(y_true, y_pred)

    RB = np.sum(y_pred - y_true) / np.sum(y_true) * 100

    return r, RMSE, MAE, RB


# %%
# site-scale ET values
le_path = r'D:\ET\TrainingDataProcess\TrainingList_Flux.csv'

df_LE = pd.read_csv(le_path)[['SiteID', 'Year', 'Month', 'EstimateLE', 'latitude', 'longitude', 'IGBP']]
df_LE['days'] = df_LE.apply(lambda x: calendar.monthrange(x['Year'], x['Month'])[1], axis=1)
df_LE['EstimateLE (mm/month)'] = df_LE['EstimateLE'] * 0.035 * df_LE['days']
# %%
hget_path = r'D:\ET\MLResults\HG-Land_ET_1982-2018_MO_ensmn.nc'
flux_path = r'D:\ET\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*'
gleam_path = r'D:\ET\DataCollect\ETProducts\Gleam\v3.6a\monthly\E_1980-2021_GLEAM_v3.6a_MO.nc'

hget = xr.open_dataset(hget_path)
flux = xr.open_mfdataset(flux_path)
gleama = xr.open_dataset(gleam_path)
# %%
for i in range(len(df_LE)):
    year, month = df_LE.at[i, 'Year'], df_LE.at[i, 'Month']
    lat, lon = df_LE.at[i, 'latitude'], df_LE.at[i, 'longitude']
    index = df_LE.index[i]
    days = df_LE.at[index, 'days']
    time = f'{str(year)}-{str(month)}'

    et_hget = hget.sel(lon=lon, lat=lat, time=time, method='nearest').LE.data
    df_LE.loc[index, 'hget (mm/month)'] = et_hget * 0.035 * days

    et_flux = flux.sel(lon=lon, lat=lat, time=time, method='nearest').compute().LE.data
    df_LE.loc[index, 'fluxet (mm/month)'] = et_flux * days / 2.45

    et_gleam = gleama.sel(lon=lon, lat=lat, time=time, method='nearest').E.data
    df_LE.loc[index, 'Gleam36aet (mm/month)'] = et_gleam

df_LE.to_csv(r'D:\ET\MLResults\compare_list.csv')
# %%
# metrics for each site
df_LE = pd.read_csv(r'D:\ET\MLResults\compare_list.csv', index_col=0)
df_LE.drop(index=df_LE[df_LE.SiteID.isin(['US-Wi1', 'US-Wi7', 'US-Wi9'])].index, inplace=True)
df_LE_group = df_LE.groupby(by=['SiteID', 'latitude', 'longitude', 'IGBP'], axis=0)
# %%
metric_dir = r'D:\ET\MLResults\Deepf_test_results'

df_LE_group.apply(lambda x: valid_metric(x['EstimateLE (mm/month)'], x['hget (mm/month)'])).to_frame().to_csv(
    metric_dir + r'\hget_metrics.csv'
)

df_LE_group.apply(
    lambda x: valid_metric(x['EstimateLE (mm/month)'], x['fluxet (mm/month)'])
).to_frame().to_csv(metric_dir + r'\flux_metrics.csv')
# %%
df_LE_gl = df_LE.dropna(axis=0, how='any')
df_LE_gp = df_LE_gl.groupby(by=['SiteID', 'latitude', 'longitude', 'IGBP'], axis=0)

df_LE_gp.apply(
    lambda x: valid_metric(x['EstimateLE (mm/month)'], x['Gleam36aet (mm/month)'])
).to_frame().to_csv(metric_dir + r'\gleamet_metrics.csv')
# %%
# metrics for each PFT type & overall metrics
df_LE = pd.read_csv(r'D:\ET\MLResults\compare_list.csv', index_col=0)
df_LE_group = df_LE.groupby(by=['IGBP'], axis=0)

df_LE_group.apply(lambda x: valid_metric(x['EstimateLE (mm/month)'], x['hget (mm/month)'])).to_frame().to_csv(
    r'D:\ET\MLResults\Deepf_test_results\hget_pft_metrics.csv'
)

df_LE_group.apply(
    lambda x: valid_metric(x['EstimateLE (mm/month)'], x['fluxet (mm/month)'])
).to_frame().to_csv(r'D:\ET\MLResults\Deepf_test_results\flux_pft_metrics.csv')
# %%
df_LE_gl = df_LE.dropna(axis=0, how='any')
df_LE_gp = df_LE_gl.groupby(by=['IGBP'], axis=0)

df_LE_gp.apply(
    lambda x: valid_metric(x['EstimateLE (mm/month)'], x['Gleam36aet (mm/month)'])
).to_frame().to_csv(r'D:\ET\MLResults\Deepf_test_results\glmaet_pft_metrics.csv')
# %%
valid_metric(y_true=df_LE['EstimateLE (mm/month)'], y_pred=df_LE['hget (mm/month)'])
valid_metric(y_true=df_LE['EstimateLE (mm/month)'], y_pred=df_LE['fluxet (mm/month)'])
valid_metric(y_true=df_LE_gl['EstimateLE (mm/month)'], y_pred=df_LE_gl['Gleam36aet (mm/month)'])
