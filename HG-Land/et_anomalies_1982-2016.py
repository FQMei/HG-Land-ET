# %%
import xarray as xr
import xcdat as xc
import numpy as np
import pandas as pd
import proplot as pplt
import statsmodels.api as sm

# %%
flux_sel = xc.open_mfdataset(
    r'D:\ET\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*.nc'
)

days = flux_sel.time.dt.daysinmonth
flux_sel_et = (flux_sel.LE * days / 2.45).compute()
# %%
weights = np.cos(np.deg2rad(flux_sel_et.lat))
flux_yr = flux_sel_et.groupby('time.year').mean(dim='time', skipna=True) * 12
flux_ts = flux_yr.weighted(weights).mean(dim=['lon', 'lat'], skipna=True)
flux_anma = flux_ts - flux_ts.mean(dim='year', skipna=True)
# %%
glma = xc.open_dataset(
    r'D:\ET\DataCollect\ETProducts\Gleam\v3.6a\yearly\E_1980-2021_GLEAM_v3.6a_YR_0.5deg.nc'
)
glma_yr = glma.E.sel(time=slice('1982', '2016'))
# %%
weights = np.cos(np.deg2rad(glma_yr.lat))
glma_ts = glma_yr.weighted(weights).mean(dim=['lon', 'lat'], skipna=True)
glma_anma = glma_ts - glma_ts.mean(dim='time', skipna=True)
# %%
hgld = xr.open_dataset(r'D:\ET\MLResults\HG-Land_ET_product\HG-Land_ET_1982-2018_MO_ensmn.nc')

hgld_sel = hgld.sel(time=slice('1982-01', '2016-12'))
days = hgld_sel.time.dt.daysinmonth
hgld_sel_et = hgld_sel.LE * 0.035 * days
# %%
weights = np.cos(np.deg2rad(hgld_sel_et.lat))
hgld_yr = hgld_sel_et.groupby('time.year').mean(dim='time', skipna=True) * 12  # mm/yr
hgld_ts = hgld_yr.weighted(weights).mean(dim=['lon', 'lat'], skipna=True)
hgld_anma = hgld_ts - hgld_ts.mean(dim='year', skipna=True)
# %%
helper = pd.DataFrame(
    data={'HG-Land': hgld_anma.values, 'FLUXCOM': flux_anma.values, 'GLEAM': glma_anma.values},
    index=np.arange(1982, 2017),
)

helper.to_csv(r'D:\ET\ETtrendCalc\glob_et_ts_1982-2016.csv')
# %%
helper = pd.read_csv(r'D:\ET\ETtrendCalc\glob_et_ts_1982-2016.csv', index_col=0)

x = np.arange(1982, 2017, 1)
X = sm.add_constant(x)
ytext = 0.3
colors = ['#EB7F00', '#1695A3', '#ACF0F2']

fig, ax = pplt.subplots(figsize=(5, 3), dpi=400)

for column, color in zip(helper.columns, colors):
    result = sm.OLS(helper[column], X).fit()

    yy = result.fittedvalues
    slope = result.params[1]
    slope_std = result.bse[1]
    p_value = result.pvalues[1]

    ax.plot(x=x, y=helper[column].values, c=color, marker="o", ms=3, lw=1.5, label=column)
    ax.plot(x=x, y=yy, c=color)

    if p_value < 0.001:
        p = 'p<0.001'
    else:
        p = f'p={p_value:.3f}'

    ax.text(
        0.5,
        ytext,
        f"slope={slope:.3f}" + "\xB1" + f"{slope_std:.3f} mm/yr$^2$, {p}",
        c=color,
        transform=ax.transAxes,
    )

    ytext -= 0.08

ax.legend()
ax.format(xlabel='Year', ylabel='ET anomalies (mm/yr)')

fig.savefig(r'D:\ET\Figures\prod_trend.png', bbox_inches='tight', dpi=400)
# %%
