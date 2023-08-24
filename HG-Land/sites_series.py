# %%
import pandas as pd
import proplot as pplt


# %%
site_dict = {
    'CRO': 'DE-Geb',
    'CSH': 'US-KS2',
    'DBF': 'US-MMS',
    'EBF': 'FR-Pue',
    'ENF': 'DE-Tha',
    'GRA': 'US-Var',
    'MF': 'BE-Bra',
    'OSH': 'US-Whs',
    'SAV': 'AU-DaS',
    'WET': 'US-Los',
    'WSA': 'US-Ton',
}

# %%
sel_df = pd.read_csv(r'D:\ET\MLResults\site_series.csv')

sel_df.date = pd.to_datetime(sel_df.date)
sel_df.set_index('date', drop=True, inplace=True)
# %%
fig, axs = pplt.subplots(ncols=3, nrows=4, figsize=(9, 8), share=1, span=True, wspace=(3.3, 3.3))
axs.format(xlabel='Year', ylabel='ET (mm/month)', abc='(a)', abcloc='l')

for pft, id, ax in zip(site_dict.keys(), site_dict.values(), axs):
    ax.scatter(y=sel_df[sel_df.SiteID == id]['EstimateLE (mm/month)'], s=3, c='k', labels='observation')

    ax.plot(
        y=sel_df[sel_df.SiteID == id]['hget (mm/month)'],
        c='#EB7F00',
        marker='o',
        ms=1.5,
        alpha=0.8,
        labels='HG-Land',
    )
    ax.plot(
        y=sel_df[sel_df.SiteID == id]['fluxet (mm/month)'],
        c='#1695A3',
        marker='o',
        ms=1.5,
        alpha=0.8,
        labels='FLUXCOM',
    )
    ax.plot(
        y=sel_df[sel_df.SiteID == id]['Gleam36aet (mm/month)'],
        c='#ACF0F2',
        marker='o',
        ms=1.5,
        alpha=0.8,
        labels='GLEAM',
    )

    ax.format(titleloc='l', title=f'{pft}: {id}', xlocator=('year', 2), xformatter='%Y', xrotation=30)

axs[-4].legend(loc='b', ncols=1, bbox_to_anchor=(0.3, -0.5, 0.3, 0.2))
axs[-1].set_visible(False)

fig.savefig(r'D:\ET\Figures\site_ts.png', bbox_inches='tight', dpi=400)
# %%
