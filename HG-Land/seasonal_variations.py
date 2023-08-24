# %%
import xarray as xr
import xcdat as xc
import numpy as np
import pandas as pd
import proplot as pplt

# %%
koppen = np.genfromtxt(r"F:\Koppen Climate map\data\koppen_1901-2010.tsv", dtype=None, names=True)

df = pd.DataFrame(koppen)
df.rename(columns={'p1901_2010': 'koppen', 'longitude': 'lon', 'latitude': 'lat'}, inplace=True)
# %%
A_list = [b'Af', b'Am', b'As', b'Aw']
B_list = [b'BSh', b'BSk', b'BWh', b'BWk']
C_list = [b'Cfa', b'Cfb', b'Cfc', b'Csa', b'Csb', b'Csc', b'Cwa', b'Cwb', b'Cwc']
D_list = [b'Dfa', b'Dfb', b'Dfc', b'Dfd', b'Dsa', b'Dsb', b'Dsc', b'Dsd', b'Dwa', b'Dwb', b'Dwc', b'Dwd']
E_list = [b'EF', b'ET']

df['koppen'][df['koppen'].isin(A_list)] = 0
df['koppen'][df['koppen'].isin(B_list)] = 1
df['koppen'][df['koppen'].isin(C_list)] = 2
df['koppen'][df['koppen'].isin(D_list)] = 3
df['koppen'][df['koppen'].isin(E_list)] = 4
df['koppen'] = df['koppen'].astype(int)
# %%
helper = pd.read_csv(r'D:\ET\TrainingDataProcess\lat_lon.csv')
helper = helper.merge(df, how='left', on=['lat', 'lon'])

ds_clas = helper.set_index(["lat", "lon"]).to_xarray()
ds_clas.to_netcdf(r'F:\Koppen Climate map\koppen_major_class.nc')
# %%
ds_clas = xr.open_dataset(r'F:\Koppen Climate map\koppen_major_class.nc')

# fluxcom
flux_sel = xc.open_mfdataset(
    r'D:\ET\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*.nc'
)
days = flux_sel.time.dt.daysinmonth
flux_sel_et = (flux_sel.LE * days / 2.45).compute()

# gleam
glma = xc.open_dataset(
    r'D:\ET\DataCollect\ETProducts\Gleam\v3.6a\monthly\E_1980-2021_GLEAM_v3.6a_MO_0.5deg.nc'
)
glma_sel_et = glma.E.sel(time=slice('1982-01', '2016-12'))

# hg-land
hgld = xr.open_dataset(r'D:\ET\MLResults\HG-Land_ET_product\HG-Land_ET_1982-2018_MO_ensmn.nc')
hgld_sel = hgld.sel(time=slice('1982-01', '2016-12'))
days = hgld_sel.time.dt.daysinmonth
hgld_sel_et = hgld_sel.LE * 0.035 * days
# %%
ds_list = [flux_sel_et, glma_sel_et, hgld_sel_et]
name_list = ['flux', 'glma', 'hgld']

for ds, file_name in zip(ds_list, name_list):
    df = pd.DataFrame(columns=['Tropical', 'Dry', 'Mild temperate', 'Snow'])

    for i in range(4):
        da_koppen = ds.where(ds_clas.koppen == i)
        arr_koppen = da_koppen.mean(dim=['lon', 'lat'], skipna=True).groupby('time.month').mean('time')
        df[df.columns[i]] = arr_koppen

    df.to_csv(f'D:\\ET\\ETprodSeasonalVar\\{file_name}_ssn_var_koppen.csv')

    print(df)
# %%
df_flux = pd.read_csv(r'D:\ET\ETprodSeasonalVar\flux_ssn_var_koppen.csv', index_col=0)
df_glma = pd.read_csv(r'D:\ET\ETprodSeasonalVar\glma_ssn_var_koppen.csv', index_col=0)
df_hgld = pd.read_csv(r'D:\ET\ETprodSeasonalVar\hgld_ssn_var_koppen.csv', index_col=0)

fig, axs = pplt.subplots(ncols=2, nrows=2, figsize=(6, 5), sharex=1, sharey=1, span=True)
axs.format(xlabel='Month', ylabel='ET (mm/month)', abc='(a)', abcloc='l')

for column, ax in zip(df_flux.columns, axs):
    ax.plot(x=np.arange(1, 13, 1), y=df_hgld[column].values, c='#EB7F00', alpha=0.6, lw=1.5, labels='HG-Land')
    ax.plot(x=np.arange(1, 13, 1), y=df_flux[column].values, c='#1695A3', alpha=0.6, lw=1.5, labels='FLUXCOM')
    ax.plot(x=np.arange(1, 13, 1), y=df_glma[column].values, c='#ACF0F2', alpha=0.6, lw=1.5, labels='GLEAM')

    ax.format(titleloc='l', title=f'{column}', xlocator=1, xtickminor=False)

axs[-1].legend(loc='ul', ncols=1, frameon=False)

fig.savefig(r'D:\ET\Figures\ssn_var.png', bbox_inches='tight', dpi=400)
# %%
