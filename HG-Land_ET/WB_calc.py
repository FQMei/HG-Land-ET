# %%

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# %%
import math
import pandas as pd
import numpy as np
import geopandas as gpd
import xarray as xr
import matplotlib.pyplot as plt
import proplot as pplt
from sklearn.metrics import r2_score, mean_squared_error
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

def test_plot(x, y, xerr, yerr, figsize, xlabel, ylabel):

    fig, ax = plt.subplots(figsize=figsize, dpi=400)

    r2 = r2_score(y, x)
    MSE = mean_squared_error(y, x)
    RMSE = math.sqrt(MSE)

    ax.errorbar(x=x, y=y, xerr=xerr, yerr=yerr, marker='.', fmt='k.',
                ms=0.5, ecolor='.3', elinewidth=0.5, capsize=1, capthick=0.3)

    plt.plot(ax.get_ylim(), ax.get_ylim(), '--', linewidth=0.8,
             color='0.5', transform=ax.transData,)

    ax.text(
        0.05, 0.75, f'$R^2$ = {r2:.2f} \nRMSE={RMSE:.2f}', transform=ax.transAxes)
    ax.set(xlabel=xlabel, ylabel=ylabel,
           xticks=[0, 500, 1000, 1500], yticks=[0, 500, 1000, 1500])
    plt.axis('square')
    return fig, ax


# %%
idall_list = [1147010, 1159100, 1196551, 1234150, 1291100, 1291200, 1531100, 1531450, 1591401, 2178950,
              2179100, 2180800, 2181900, 2186800, 2260500, 2903420, 2906900, 2909150, 2912600, 2998510,
              2999110, 2999910, 3265601, 3275750, 3275990, 3618500, 3621200, 3623100, 3633120, 3634340,
              3635030, 3635040, 3635650, 3637150, 3638050, 3649950, 3650481, 3651807, 4103200, 4115201,
              4127800, 4143550, 4150450, 4150500, 4151804, 4152050, 4207900, 4208025, 4209805, 4213711,
              4214025, 4214051, 4214270, 4214520, 4244500, 4355100, 5101200, 5101301, 5404270, 6279500,
              6340110, 6357010, 6435060, 6458010, 6590700, 6742900, 6955430, 6970250, 6970700, 6977100,
              6978250,
              1901, 1902, 1905, 1908, 1911, 1912, 1913, 1914, 2902, 2905,
              2906, 2908, 2909, 2910, 2912, 2914, 2915, 3908, 4901, 5902, 5904,
              ]
# %%
shp_path = r'D:\ET_FQM\DataCollect\Runoff\GRDC\GRDC_3degree_from2003\stationbasins.shp'
mrb_path = r'D:\ET_FQM\DataCollect\Runoff\GRDC\mrb_shp_zip\mrb_basins.shp'

basin_shp = gpd.read_file(shp_path)
mrb_shp = gpd.read_file(mrb_path)
# %%
# ET products

# hget
ET_path = r'D:\ET_FQM\ETsubject\MLResults\\HG-Land_ET_1982-2018_MO.nc'
ET_clip_dir = r'D:\ET_FQM\ETsubject\BasinCalc\HGET_process\et4basins_hget'
ET_ann_dir = r'D:\ET_FQM\ETsubject\BasinCalc\HGET_process\hget_ann'

# fluxcom
# ET_path = r'D:\ET_FQM\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*.nc'
# ET_clip_dir = r'D:\ET_FQM\ETsubject\BasinCalc\Fluxcom_process\et4basins_fluxcom'
# ET_ann_dir = r'D:\ET_FQM\ETsubject\BasinCalc\Fluxcom_process\fluxcom_ann'

# gleam
# ET_path = r'D:\ET_FQM\DataCollect\ETProducts\Gleam\v3.6a\yearly\E_1980-2021_GLEAM_v3.6a_YR.nc'
# ET_clip_dir = r'D:\ET_FQM\ETsubject\BasinCalc\Gleam_process\et4basins_gleam'
# ET_ann_dir = r'D:\ET_FQM\ETsubject\BasinCalc\Gleam_process\gleam_ann'
# %%
# hget
ds_et = xr.open_dataset(ET_path).sel(time=slice('2003-01-01', '2016-12-31'))

# fluxcom
# ds_et = xr.open_mfdataset(ET_path).sel(time=slice('2003-01-01', '2016-12-31'))

# gleam
# gleama = xr.open_dataset(ET_path).sel(time=slice('2003-12-31', '2016-12-31'))
# ds_et = gleama.where(gleama.E > -999, np.nan)
# %%
for id in idall_list:

    if id > 9999:
        # etmask = ds_et.LE.salem.roi(shape=basin_shp[basin_shp.grdc_no == id])
        etmask = ds_et.salem.roi(shape=basin_shp[basin_shp.grdc_no == id])
    else:
        # etmask = ds_et.LE.salem.roi(shape=mrb_shp[mrb_shp.MRBID == id])
        etmask = ds_et.salem.roi(shape=mrb_shp[mrb_shp.MRBID == id])
    
    etmask.to_netcdf(ET_clip_dir+f'\\et_{id}.nc')

    weights = np.cos(np.deg2rad(etmask.lat))

    # HGET
    et_ann = etmask.LE.weighted(weights).mean(
        dim=['lon', 'lat'], skipna=True).groupby('time.year').mean()*12.876 
    df_ann = et_ann.to_dataframe().rename(columns={'LE': 'ET_ml (mm/yr)'})
    df_ann.to_csv(ET_ann_dir+f'\\et_{id}.csv')

    # FLUXCOM
    # et_ann = etmask.LE.weighted(weights).mean(
    #     dim=['lon', 'lat'], skipna=True).groupby('time.year').mean()*365.25/2.45
    # df_ann = et_ann.to_dataframe().rename(columns={'LE': 'ET_ml (mm/yr)'})
    # df_ann.to_csv(ET_ann_dir+f'\\et_{id}.csv')

    # GLEAM
    # df_ann = etmask.E.weighted(weights).mean(dim=['lon', 'lat'], skipna=True).to_dataframe().rename(columns={'E': 'ET_ml (mm/yr)'})
    # df_ann.to_csv(ET_ann_dir+f'\\et_{id}.csv')
# %%
Pyr_dir = r'D:\ET_FQM\ETsubject\BasinCalc\GPMprocess\gpm_ann'
Qyr_dir = r'D:\ET_FQM\ETsubject\BasinCalc\GRDCprocess\grdc_ann'
TWSCyr_dir = r'D:\ET_FQM\ETsubject\BasinCalc\TWSCprocess\twsc_ann'
# %%
# HGET
ETyr_dir = r'D:\ET_FQM\ETsubject\BasinCalc\HGET_process\hget_ann'
wb_ann_path = r'D:\ET_FQM\ETsubject\BasinCalc\HGET_process\wb_ann_hget.csv'
wb_annmean_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\FigureA\hget_wb_annmean.png'

# Fluxcom
# ETyr_dir = r'D:\ET_FQM\ETsubject\BasinCalc\Fluxcom_process\fluxcom_ann'
# wb_ann_path = r'D:\ET_FQM\ETsubject\BasinCalc\Fluxcom_process\wb_ann_fluxcom.csv'
# wb_annmean_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\FigureA\fluxcom_wb_annmean.png'

# Gleam
# ETyr_dir = r'D:\ET_FQM\ETsubject\BasinCalc\Gleam_process\gleam_ann'
# wb_ann_path = r'D:\ET_FQM\ETsubject\BasinCalc\Gleam_process\wb_ann_gleam.csv'
# wb_annmean_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\FigureA\gleam_wb_annmean.png'
# %%
inland_basin_path = r'D:\ET_FQM\ETsubject\BasinCalc\GRDCprocess\inland_basin_3degree_from2003.csv'
grdc_station_path = r'D:\ET_FQM\DataCollect\Runoff\GRDC\grdc_stationslist\GRDC_stations.xlsx'

df_inbsn = pd.read_csv(inland_basin_path)
df_sta = pd.read_excel(grdc_station_path, 'station_catalogue')
# %%
helper = pd.DataFrame(columns=['id', 'area (km2)', 'year', 
                                'P (mm/yr)', 'Q (mm/yr)', 'twsc (mm/yr)',
                                'LE', 'ET_wb (mm/yr)'])

for id in idall_list:

    if id > 9999:
        Q = pd.read_csv(Qyr_dir+f'\\grdc_{id}.csv')[['Q (mm/yr)']]
        area = df_sta['area'][df_sta.grdc_no == id].values[0]
    else:
        Q = pd.DataFrame(data={'Q (mm/yr)': [0]*14})
        area = df_inbsn['SUM_SUB_AREA'][df_inbsn.MRBID == id].values[0]

    P = pd.read_csv(Pyr_dir+f'\\gpm_{id}.csv')[['year', 'P (mm/yr)']]

    TWSC = pd.read_csv(TWSCyr_dir+f'\\twsc_{id}.csv')[['twsc (mm/yr)']]

    ET = pd.read_csv(ETyr_dir+f'\\et_{id}.csv')[['ET_ml (mm/yr)']]

    df_con = pd.concat([P, Q, TWSC, ET], axis=1, join='outer')
    df_con.insert(loc=0, column='id', value=id)
    df_con.insert(loc=1, column='area (km2)', value=area)

    df_con['ET_wb (mm/yr)'] = df_con['P (mm/yr)'] - \
        df_con['Q (mm/yr)'] - df_con['twsc (mm/yr)']

    helper = pd.concat([helper, df_con])

helper.to_csv(wb_ann_path, index=False)

helper
# %%
et4com = helper[['id', 'area (km2)', 'ET_wb (mm/yr)','LE']].copy()

et4com[et4com < 0] = np.nan
et4com.dropna(inplace=True)

et4com['LE'], et4com['ET_wb (mm/yr)'] = et4com['LE'].astype(
    float), et4com['ET_wb (mm/yr)'].astype(float)
# %%
et4com_path = r'D:\ET_FQM\ETsubject\BasinCalc\Gleam_process\gleam_mean_std_3degree.csv'

et4com_mean_ann = et4com.groupby(['id', 'area (km2)']).agg(
    {'ET_ml (mm/yr)': ['mean', 'std'], 'ET_wb (mm/yr)': ['mean', 'std']})
et4com_mean_ann.to_csv(et4com_path)

et4com_mean_ann.fillna(0, inplace=True)
et4com_mean_ann.reset_index(inplace=True)
# %%
wb_fig_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\hget_annual_mean.png'

xlabel = 'ET$_{HGET}$ (mm yr$^{-1}$)'

fig, ax = test_plot(x=et4com_mean_ann.loc[:, ('LE', 'mean')],
                    y=et4com_mean_ann.loc[:, ('ET_wb (mm/yr)', 'mean')],
                    xerr=et4com_mean_ann.loc[:, ('LE', 'std')],
                    yerr=et4com_mean_ann.loc[:, ('ET_wb (mm/yr)', 'std')],
                    xlabel=xlabel,
                    ylabel='ET$_{WB}$ (mm yr$^{-1}$)',
                    figsize=(2, 2))

fig.savefig(wb_fig_path, bbox_inches='tight', dpi=400)
# %%
