# %%
import pandas as pd
import proplot as pplt
import numpy as np

# %%
metric_dir = r'D:\ET\MLResults\Deepf_test_results'
metric_hget = pd.read_csv(metric_dir + r'\hget_metrics.csv')
# %%
columns = ['r', 'RMSE(mm/mon)', 'MAE(mm/mon)', 'RB(%)']
ylabels = ['R', 'RMSE (mm/month)', 'MAE (mm/month)', 'RB (%)']
cycles = (
    'yellow5',
    'earth',
    'umber',
    'muted green',
    'emerald green',
    'kelly green',
    'grass',
    'mud green',
    'chartreuse',
    'yellow green',
    'sky blue',
)

igbp_list = ['CRO', 'CSH', 'OSH', 'DBF', 'EBF', 'ENF', 'MF', 'GRA', 'SAV', 'WSA', 'WET']

columns = ['r', 'RMSE(mm/mon)', 'MAE(mm/mon)', 'RB(%)']
ylabels = ['R', 'RMSE (mm/month)', 'MAE (mm/month)', 'RB (%)']

lower_labels = metric_hget.pivot(columns='IGBP', values='r').count(axis=0)

fig, axs = pplt.subplots(ncols=2, nrows=2, figsize=(8, 5), share=False)
axs.format(abc='(a)', abcloc='l')

for ax, value, ylabel in zip(axs, columns, ylabels):
    ax.box(
        metric_hget.pivot(columns='IGBP', values=value)[igbp_list],
        alpha=0.4,
        marker='.',
        lw=0.8,
        widths=0.6,
        cycle=cycles,
        showfliers=False,
    )
    ax.format(xlabel='', ylabel=ylabel)

    for i in np.arange(11):
        ax.text(
            i,
            0.03,
            lower_labels[i],
            transform=ax.get_xaxis_transform(),
            horizontalalignment='center',
            size='small',
            color=cycles[i],
        )

    fig.savefig(f'D:\\ET\\Figures\\hget_pfts.png', bbox_inches='tight', dpi=400)
# %%
