# %%
import pandas as pd
import proplot as pplt

# %%
pft_metric_path = r'D:\ET\MLResults\Deepf_test_results\pft_metris_all.xlsx'
df_r = pd.read_excel(pft_metric_path, sheet_name='R', index_col=0)
df_rmse = pd.read_excel(pft_metric_path, sheet_name='RMSE', index_col=0)
df_mae = pd.read_excel(pft_metric_path, sheet_name='MAE', index_col=0)
df_rb = pd.read_excel(pft_metric_path, sheet_name='RB', index_col=0)
# %%
ylabels = ['R', 'RMSE (mm/month)', 'MAE (mm/month)', 'RB (%)']
df_list = [df_r, df_rmse, df_mae, df_rb]
cycle = ('#CEC3D1', '#87C1B8', '#FEE3A3')

fig, axs = pplt.subplots(ncols=2, nrows=2, figsize=(9, 5), share=False, dpi=400)
axs.format(abc='(a)', abcloc='l')

for ax, df, ylabel in zip(axs, df_list, ylabels):
    if ylabel != 'RB (%)':
        ax.bar(df, cycle=cycle, alpha=1, edgecolor='gray')
    else:
        ax.bar(
            df, cycle=cycle, legend='ur', legend_kw={'frame': False, 'ncols': 1}, alpha=1, edgecolor='gray'
        )
    ax.format(ylabel=ylabel)

fig.savefig(r'D:\ET\Figures\pfts_metrics.png', bbox_inches='tight', dpi=400)
# %%
