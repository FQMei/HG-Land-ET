# %%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# %%
igbp_list = ["CRO", "CSH", "DBF", "EBF", "ENF", "GRA", "MF", "OSH", "SAV", "WET", "WSA"]

for igbp in igbp_list:
    ds = xr.open_dataset(
        f"D:\\ET\\MLResults\\HG-Land_ET_product\\HG-Land_ET_1982-2018_MO_{igbp}.nc"
    )
    ds_ann = ds.LE.mean(dim="time", skipna=True) * 12.876

    ds_ann.to_netcdf(
        f"D:\\ET\\MLResults\\HG-Land_ET_product\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_{igbp}.nc"
    )
# %%
ds_path = r"D:\ET\MLResults\HG-Land_ET_product\"

ds_cro = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_CRO.nc")
ds_csh = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_CSH.nc")
ds_dbf = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_DBF.nc")
ds_ebf = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_EBF.nc")
ds_enf = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_ENF.nc")
ds_gra = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_GRA.nc")
ds_mf = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_MF.nc")
ds_osh = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_OSH.nc")
ds_sav = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_SAV.nc")
ds_wet = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_WET.nc")
ds_wsa = xr.open_dataset(ds_path + "\\HG-Land_ET_1982-2018_multiyr_mean_mmyr_WSA.nc")
# %%
ds_stack = [
    ds_cro.LE.data,
    ds_csh.LE.data,
    ds_dbf.LE.data,
    ds_ebf.LE.data,
    ds_enf.LE.data,
    ds_gra.LE.data,
    ds_mf.LE.data,
    ds_osh.LE.data,
    ds_sav.LE.data,
    ds_wet.LE.data,
    ds_wsa.LE.data,
]
# %%
ET_arr = np.stack(ds_stack, axis=0)
ET_std = np.std(ET_arr, axis=0, ddof=1)
# %%
ds_std = xr.Dataset(
    data_vars={"LE_std": (("lat", "lon"), ET_std)},
    coords={"lat": ds_cro.lat, "lon": ds_cro.lon},
)
# %%
fig, ax = plt.subplots(
    figsize=(5, 3.5), dpi=500, subplot_kw={"projection": ccrs.PlateCarree()}
)

std = Spatial_map(ax, da=ds_std.LE_std, norm=None, cmap="Spectral_r")

cb = plt.colorbar(
    std,
    ax=ax,
    shrink=0.8,
    aspect=22,
    pad=0.10,
    extend="max",
    location="bottom",
    label="Uncertainty in ET estimates (mm/yr)",
)

cb.ax.tick_params(
    which="both",
    direction="in",
)

fig.savefig(
    r"D:\ET\Figures\uncertainty_multiyr_mean_1982-2018.tiff",
    bbox_inches="tight",
    dpi=500,
)