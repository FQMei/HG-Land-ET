# %%
import pandas as pd
import proplot as pplt

# %%
etbsn_path = r'D:\ET\BasinCalc\wb_closure'

df_clas = pd.read_csv(etbsn_path + r'\class_mrb_et.csv', index_col=0)
df_flux = pd.read_csv(etbsn_path + r'\flux_mrb_et.csv', index_col=0)
df_glma = pd.read_csv(etbsn_path + r'\glma_mrb_et.csv', index_col=0)
df_hget = pd.read_csv(etbsn_path + r'\hget_mrb_et.csv', index_col=0)
# %%
for df in [df_clas, df_flux, df_glma, df_hget]:
    df.index = pd.to_datetime(df.index)
# %%
df_mrb = pd.read_csv(r'D:\ET\DataCollect\Runoff\GRDC\mrb_sel_shp\mrb_info.csv', index_col=0)
# %%
id_list = ['4219', '3203', '1209', '6903', '2433', '5902']

fig, axs = pplt.subplots(ncols=2, nrows=3, figsize=(9, 8), share=1, span=True)
axs.format(xlabel='Year', ylabel='ET (mm/month)', abc='(a)', abcloc='l')

for id, ax in zip(id_list, axs):
    title = [df_mrb[df_mrb.MRBID == int(id)].iat[0, 1] if id != '3203' else 'AMAZON'][0]

    ax.scatter(y=df_clas[id], s=3, c='k', labels='CLASS')

    ax.plot(y=df_hget[id], c='#EB7F00', marker='o', ms=1.5, alpha=0.8, labels='HG-Land')
    ax.plot(y=df_flux[id], c='#1695A3', marker='o', ms=1.5, alpha=0.8, labels='FLUXCOM')
    ax.plot(y=df_glma[id], c='#ACF0F2', marker='o', ms=1.5, alpha=0.8, labels='GLEAM')

    ax.format(titleloc='l', title=title, xlocator=('year',), xformatter='%Y', xrotation=30)

axs[-1].legend(loc='ul', ncols=2, frame=False)

fig.savefig(r'D:\ET\Figures\mrb_ts.png', bbox_inches='tight', dpi=400)
