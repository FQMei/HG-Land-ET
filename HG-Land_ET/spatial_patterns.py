# %%
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# %%
import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import proplot as pplt
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
hget_path = r'D:\ET_FQM\ETsubject\MLResults\HG-Land_ET_1982-2018_MO.nc'
flux_path = r'D:\ET_FQM\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*.nc'
gleam_path = r'D:\ET_FQM\DataCollect\ETProducts\Gleam\v3.6a\yearly\E_1980-2021_GLEAM_v3.6a_YR.nc'
# %%
hget = xr.open_dataset(hget_path)
flux = xr.open_mfdataset(flux_path)
gleam = xr.open_dataset(gleam_path)
# %%
hget_ann = hget.LE.sel(time=slice(
    '1982-01', '2016-12')).mean(dim='time', skipna=True)*12.876  
# %%
flux_ann = flux.LE.mean(dim='time', skipna=True)*365.25/2.45
# %%
gleam_ann = gleam.E.sel(time=slice(
    '1982-01', '2016-12')).mean(dim='time', skipna=True)

gleam_ann_intp = gleam_ann.interp(lon=flux.lon.values, lat=flux.lat.values)
# %%
hget_ann_mask = hget_ann.where(~np.isnan(flux_ann))
gleam_ann_mask = gleam_ann_intp.where(~np.isnan(flux_ann))
# %%


def Spatial_map(ax, da, norm, cmap):

    ax.set_global()
    ax.coastlines(linewidth=0.4, alpha=0.8)
    ax.add_feature(cfeature.BORDERS.with_scale(
        '110m'), linewidth=0.4, alpha=0.8)
    ax.add_feature(cfeature.LAND, fc='gray5', alpha=0.3, zorder=0)
    ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())

    conf = da.plot.pcolormesh(
        darray=da,
        ax=ax,
        add_colorbar=False,
        cmap=cmap,
        norm=norm,
        zorder=1,
        transform=ccrs.PlateCarree(),
    )

# %%


def colorbar_plot(norm, cmap, ticks, label, extend='both'):

    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                      orientation='horizontal',
                      shrink=0.65, cax=ax, drawedges=False,
                      aspect=35,
                      ticks=ticks,
                      extend=extend,
                      )

    cb.ax.tick_params(which='major', direction='in', 
                      width=0.4, length=2.5, labelsize=8)

    cb.ax.set_xlabel(xlabel=label, labelpad=0.02)
    cb.outline.set_linewidth(0.4)

# %%
# spatial pattern
da = [hget_ann_mask, flux_ann, gleam_ann_mask]
figname = ['hget', 'fluxcru', 'gleam36a']

map1 = plt.cm.terrain_r(np.linspace(0, 1, 256))
map2 = plt.cm.Oranges(np.linspace(0.5, 1, 256))
all_maps = np.vstack((map2, map1))
cmap_custom = colors.LinearSegmentedColormap.from_list('cmap_custom', all_maps)
# %%
norm_spatial = colors.TwoSlopeNorm(vmin=-5, vcenter=0, vmax=1600)

for i in range(3):

    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                           figsize=(5, 3), dpi=300)

    Spatial_map(ax, da=da[i], norm=norm_spatial, cmap=cmap_custom)
    plt.axis('off')

    plt.savefig(f'D:\\ET_FQM\\ETsubject\\Analysis\\Figures\\Annual_1982-2016_{figname[i]}.eps',
                dpi=300, bbox_inches='tight')
# %%
fig, ax = plt.subplots(figsize=(2, 1))
fig.subplots_adjust(bottom=0.8,)

colorbar_plot(norm=norm_spatial, cmap=cmap_custom, 
              ticks=np.arange(0, 1601, 400), 
              label='ET (mm yr$^{-1}$)', extend='both')

fig.savefig(r'D:\ET_FQM\ETsubject\Analysis\Figures\colorbar_annual.png',
            bbox_inches='tight', dpi=300)
# %%
# diff
diff = [da[0]-da[1], da[0]-da[2], da[1]-da[2]]
figname = ['hget-flux', 'hget-gleam', 'flux-gleam']

norm_diff = colors.Normalize(vmin=-600, vmax=600)
# %%
for i in range(3):
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()},
                           figsize=(5, 3), dpi=300)

    Spatial_map(ax=ax, da=diff[i], norm=norm_diff, cmap='BrBG')
    plt.axis('off')

    fig.savefig(f'D:\\ET_FQM\\ETsubject\\Analysis\\Figures\\Diff_{figname[i]}.png',
                bbox_inches='tight', dpi=300)
# %%
fig, ax = plt.subplots(figsize=(2, 1))
fig.subplots_adjust(bottom=0.8)

colorbar_plot(norm=norm_diff, cmap='BrBG', 
              ticks=np.arange(-600, 601, 300), 
              label='Difference (mm yr$^{-1}$)', extend='both')

fig.savefig(r'D:\ET_FQM\ETsubject\Analysis\Figures\colorbar_annual_diff.png',
            bbox_inches='tight', dpi=300)
# %%
colors = ['#ef767a', '#456990', '#49beaa']
labels = ['HG-Land ET', 'FLUXCOM', 'GLEAM']

fig, axes = plt.subplots(2, 1, figsize=(3, 2))
plt.subplots_adjust(hspace=0.25)

for i in range(3):
    Lat_annual = da[i].mean(dim=['lon'], skipna=True)
    Lon_annual = da[i].mean(dim=['lat'], skipna=True)

    axes[0].plot(Lat_annual.lat.data, Lat_annual.data,
                 c=colors[i], lw=1, zorder=1, label=labels[i])

    axes[0].set(xticks=np.arange(-60, 91, 30), ylabel='ET (mm yr$^{-1}$)')
    axes[0].xaxis.set_major_formatter(LatitudeFormatter())

    axes[0].spines[['bottom', 'top', 'left', 'right']].set_linewidth(1)
    axes[0].tick_params(width=0.8, length=2)
    axes[0].legend(frameon=False)

    axes[1].plot(Lon_annual.lon.data, Lon_annual.data,
                 c=colors[i], lw=1, zorder=1)
    axes[1].set(xticks=np.arange(-180, 181, 60), ylabel='ET (mm yr$^{-1}$)')
    axes[1].xaxis.set_major_formatter(LongitudeFormatter())

    axes[1].spines[['bottom', 'top', 'left', 'right']].set_linewidth(1)
    axes[1].tick_params(width=0.8, length=2)

fig.savefig(r'D:\ET_FQM\ETsubject\Analysis\Figures\long_lati_distibution.png',
            bbox_inches='tight', dpi=200)
# %%
