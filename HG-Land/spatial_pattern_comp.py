# %%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# %%
hget_path = r'D:\ET\MLResults\HG-Land_ET_1982-2018_MO_ensmn.nc'
flux_path = r'D:\ET\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*.nc'
gleam_path = r'D:\ET\DataCollect\ETProducts\Gleam\v3.6a\yearly\E_1980-2021_GLEAM_v3.6a_YR_0.5deg.nc'
# %%
hget = xr.open_dataset(hget_path)
flux = xr.open_mfdataset(flux_path)
gleam = xr.open_dataset(gleam_path)
# %%
hget_ann = hget.LE.sel(time=slice('1982-01', '2016-12')).mean(dim='time', skipna=True) * 12.876

flux_ann = flux.LE.mean(dim='time', skipna=True) * 365.25 / 2.45

gleam_ann = gleam.E.sel(time=slice('1982-01', '2016-12')).mean(dim='time', skipna=True)
# %%
hget_ann_mask = hget_ann.where(~np.isnan(flux_ann))
gleam_ann_mask = gleam_ann.where(~np.isnan(flux_ann))
# %%


def Spatial_map(ax, da, norm, cmap):
    ax.set_global()
    ax.coastlines(linewidth=0.4, alpha=0.8)
    ax.add_feature(cfeature.LAND, fc='gray5', alpha=0.3, zorder=0)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    conf = da.plot.imshow(
        ax=ax, add_colorbar=False, cmap=cmap, norm=norm, zorder=1, transform=ccrs.PlateCarree()
    )

    return conf


# %%
norm_spatial = colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=1600)
norm_diff = colors.Normalize(vmin=-600, vmax=600)

map1, map2 = plt.cm.terrain_r(np.linspace(0, 1, 256)), plt.cm.binary(np.linspace(0.5, 1, 256))
all_maps = np.vstack((map2, map1))
cmap_custom = colors.LinearSegmentedColormap.from_list('cmap_custom', all_maps)
# %%
color_list = ['#ef767a', '#456990', '#49beaa']
labels = ['HG-Land', 'FLUXCOM', 'GLEAM']
da = [hget_ann_mask, hget_ann_mask - flux_ann, hget_ann_mask - gleam_ann_mask]

# %%
fig = plt.figure(figsize=(8, 5.5), dpi=500)

gs = fig.add_gridspec(3, 3)

ax1 = fig.add_subplot(gs[0, :2], projection=ccrs.PlateCarree())
conf = Spatial_map(ax1, da=da[0], norm=norm_spatial, cmap=cmap_custom)
ax1.text(0.05, 0.1, '(a)', transform=ax1.transAxes)

ax2 = fig.add_subplot(gs[1, :2], projection=ccrs.PlateCarree())
diff = Spatial_map(ax2, da=da[1], norm=norm_diff, cmap='BrBG')
ax2.text(0.05, 0.1, '(b)', transform=ax2.transAxes)

ax3 = fig.add_subplot(gs[2, :2], projection=ccrs.PlateCarree())
diff = Spatial_map(ax3, da=da[2], norm=norm_diff, cmap='BrBG')
ax3.text(0.05, 0.1, '(c)', transform=ax3.transAxes)

ax4 = fig.add_subplot(gs[:, 2])
for i in range(3):
    Lat_annual = da[i].mean(dim=['lon'], skipna=True)
    ax4.plot(Lat_annual.data, Lat_annual.lat.data, c=color_list[i], lw=1, alpha=1, zorder=1, label=labels[i])

    ax4.set(yticks=np.arange(-60, 91, 30), xlabel='ET (mm/yr)')
    ax4.yaxis.set_major_formatter(LatitudeFormatter())

    ax4.legend(frameon=False)
    ax4.yaxis.set_label_position('right')
    ax4.yaxis.tick_right()

plt.subplots_adjust(wspace=0.1, hspace=0.2)

cb1 = plt.colorbar(
    conf, ax=[ax1], shrink=1, aspect=15, pad=0.1, extend='both', location='left', label='ET (mm/yr)'
)
cb1.ax.tick_params(which='major', direction='out', width=0.5, length=1)

cb2 = plt.colorbar(
    diff,
    ax=[ax2, ax3],
    shrink=1,
    aspect=30,
    pad=0.1,
    extend='both',
    location='left',
    label='Difference (mm/yr)',
)
cb2.ax.tick_params(which='major', direction='out', width=0.5, length=1)

plt.savefig(f'D:\\ET\\Figures\\Annual_1982-2016.png', dpi=500, bbox_inches='tight')
# %%
