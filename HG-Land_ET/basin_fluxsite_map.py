#!/usr/bin/env python
# -*- encoding: utf-8 -*-

# %%
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import numpy as np
import pandas as pd
import proplot as pplt
import geopandas as gpd
# %%
params = {
    'lines.linewidth': 1, 'font.family': 'sans-serif',
    'font.sans-serif': 'Helvetica', 'font.size': 9,
    'axes.titlesize': 10, 'axes.labelsize': 10,
    'tick.dir': 'in', 'tick.labelpad': 1,
    'tick.len': 3, 'tick.width': 0.8,
    'tick.labelsize': 10, 'tick.minor': False,
    'legend.fontsize': 9,
    'label.pad': 3, 'label.size': 10,
    'grid': False, 'grid.linestyle': '--',
    'grid.linewidth': 0.1, 'gridminor': False,
    'meta.width': 1,
}
pplt.rc.update(params)
# %%
sample_path = r'D:\ET_FQM\ETsubject\DataProcess\TrainingList_Flux.xlsx'
siteinfo_path = r'D:\ET_FQM\ETsubject\DataProcess\FluxnetSiteInfo.xlsx'
basinshp_path = r'D:\ET_FQM\DataCollect\Data_wb\GRDC\GRDC_3degree_from2003\stationbasins.shp'
mrbshp_path = r'D:\ET_FQM\DataCollect\Data_wb\GRDC\mrb_shp_zip\mrb_basins.shp'
# %%
basinshp = gpd.read_file(basinshp_path)
mrbshp = gpd.read_file(mrbshp_path)
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
# %%
inland_list = [1901, 1902, 1905, 1908, 1911, 1912, 1913, 1914, 2902, 2905,
               2906, 2908, 2909, 2910, 2912, 2914, 2915, 3908, 4901, 5902, 5904]

basinshp_ = basinshp.drop(index=[4, 60])
mrbshp_ = mrbshp[mrbshp['MRBID'].isin(inland_list)]
world_ = world[world.name != 'Antarctica']
# %%
df_sample = pd.read_excel(io=sample_path, sheet_name='Sheet1')
df_siteinfo = pd.read_excel(io=siteinfo_path, sheet_name='Sheet1')[
    ['SiteID', 'Country', 'Latitude', 'Longitude', 'IGBP']]
# %%
unique_sites = df_sample.SiteID.unique()
df_sites = df_siteinfo[df_siteinfo.SiteID.isin(unique_sites)].reset_index(drop=True)
# %%
geometry = gpd.points_from_xy(df_sites.Longitude, df_sites.Latitude)
geo_df = gpd.GeoDataFrame(df_sites, geometry=geometry)

geo_df.head()
# %%
fig, ax = pplt.subplots(figsize=(5, 3),proj='cyl')

ax.format(land=True, landcolor='navy',labels=False)

basinshp_.plot(ax=ax, column='area_hys',  cmap='YlOrRd', alpha=0.8,zorder=3)
mrbshp_.plot(ax=ax, column='SUM_SUB_AR', cmap='YlOrRd', alpha=0.8,zorder=3)

basinshp_.boundary.plot(ax=ax, color='white', linewidth=.1,zorder=3)
mrbshp_.boundary.plot(ax=ax, color='white', linewidth=.1,zorder=3)

geo_df.plot(column='IGBP', cmap='gist_rainbow',
            ax=ax, legend=True, markersize=5,
            legend_kwds={'loc': 'b','ncols':6,'frameon':False},
            zorder=4,
            )

ax.set_xticks(ticks=np.arange(-180, 181, 60))
ax.set_yticks(ticks=np.arange(-90, 91, 45))
ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
ax.yaxis.set_major_formatter(LatitudeFormatter())

fig.savefig(r'D:\ET_FQM\ETsubject\Analysis\Figures\basinsite_map.png',
            bbox_inches='tight', dpi=200)
# %%
