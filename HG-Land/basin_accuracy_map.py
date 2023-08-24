# %%
import pandas as pd
import numpy as np
import geopandas as gpd
import proplot as pplt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# %%
def metric_plot(fig, ax, mrbshp_, metric_name, vmin, vmax, title, cmap):
    cax = fig.add_axes([0.58, 0.25, 0.2, 0.03])
    cax.tick_params(which='both', labelsize=8, direction='in')

    m = mrbshp_.plot(
        ax=ax,
        column=metric_name,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        legend=True,
        legend_kwds={"orientation": "horizontal", 'pad': 0.1, 'extend': 'both', 'aspect': 30},
        cax=cax,
        alpha=0.8,
        zorder=3,
    )
    mrbshp_.boundary.plot(ax=ax, color='gray7', linewidth=0.1, zorder=3)

    ax.format(
        land=True,
        ocean=True,
        latlim=(-60, 90),
        landcolor='gray3',
        lonlines=60,
        latlines=30,
        titleloc='l',
        title=title,
    )

    ax.set_xticks(ticks=np.arange(-120, 180, 60))
    ax.set_yticks(ticks=np.arange(-60, 91, 30))
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())

    ix = ax.inset_axes([0.08, 0.13, 0.17, 0.3], transform='axes', proj='cart')
    ix.format(xlabel='', ticklabelsize=8)
    ix.hist(
        mrbshp_[metric_name],
        bins=20,
        weights=np.zeros_like(mrbshp_[metric_name]) + 1 / len(mrbshp_[metric_name]),
        filled=True,
        alpha=0.3,
        c='red5',
    )


# %%
mrbshp = gpd.read_file(r'D:\ET\DataCollect\Runoff\GRDC\mrb_sel_shp_QM\mrb_sel.shp')

metric_hget = pd.read_csv(r'D:\ET\BasinCalc\wb_closure\hget_metric.csv')
metric_flux = pd.read_csv(r'D:\ET\BasinCalc\wb_closure\flux_metric.csv')
metric_glma = pd.read_csv(r'D:\ET\BasinCalc\wb_closure\glma_metric.csv')
# %%
mrbshp_hget = pd.merge(mrbshp, metric_hget, on='MRBID')
mrbshp_flux = pd.merge(mrbshp, metric_flux, on='MRBID')
mrbshp_glma = pd.merge(mrbshp, metric_glma, on='MRBID')
# %%
mrbshp_list = [mrbshp_hget, mrbshp_flux, mrbshp_glma]
filename_list = ['hget', 'flux', 'glma']

titles = ['(a) R', '(b) RMSE (mm/month)', '(c) MAE (mm/month)', '(d) RB (%)']
vmins = [-1, 0, 0, -100]
vmaxs = [1, 40, 30, 100]
columns = ['r', 'RMSE(mm/mon)', 'MAE(mm/mon)', 'RB(%)']

for mrbshp_, file_name in zip(mrbshp_list, filename_list):
    for column, vmin, vmax, title in zip(columns, vmins, vmaxs, titles):
        fig, ax = pplt.subplots(figsize=(4, 2.4), proj='cyl')

        cmap = ['terrain_r' if column == 'r' else 'terrain'][0]

        metric_plot(
            fig=fig, ax=ax, mrbshp_=mrbshp_, metric_name=column, vmin=vmin, vmax=vmax, title=title, cmap=cmap
        )

        fig.savefig(f'D:\\ET\\Figures\\{file_name}_basn_{column[:3]}.png', bbox_inches='tight', dpi=400)
