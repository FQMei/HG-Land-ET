# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import math


# %%
def PM_ET0(Rn, G, Tmean, U10, VPD, P):
    Rn = Rn * 0.0864
    G = G * 0.0864
    VPD = VPD * 0.1

    delta = 4098 * 0.6108 * np.exp(17.27 * Tmean / (Tmean + 237.3)) / ((Tmean + 237.3) ** 2)
    r = 0.665 / 1000 * P
    U2 = U10 * np.log(128) / np.log(661.3)

    ET0_1 = 0.408 * delta * (Rn - G)
    ET0_2 = r * 900 / (Tmean + 273.16) * U2 * VPD
    ET0_3 = delta + r * (1 + 0.34 * U2)
    pet = (ET0_1 + ET0_2) / ET0_3

    return pet


# %%
def test_plot(x, y, color, xlabel, ylabel, xtext, ytext, label, ax):
    MSE = mean_squared_error(y, x)
    RMSE = math.sqrt(MSE)

    r = pearsonr(y, x)[0]

    MAE = mean_absolute_error(y, x)

    RB = np.sum(x - y) / np.sum(y) * 100

    ax.scatter(x=x, y=y, s=4, marker='o', alpha=0.2, color=color, label=label)
    ax.plot(ax.get_ylim(), ax.get_ylim(), '--', linewidth=0.8, color='0.5', transform=ax.transData)

    ax.text(
        x=xtext,
        y=ytext,
        s=f'$R$ = {r:.2f}\nRMSE = {RMSE:.2f}\nMAE = {MAE:.2f}\nRB = {RB:.2f}',
        c=color,
        transform=ax.transAxes,
    )

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.set_yticks([0, 100, 200, 300])
    ax.set_aspect('equal', adjustable='box')
    ax.legend()


# %%
df_train = pd.read_csv(r'D:\ET\TrainingDataProcess\TrainingList_Flux.csv')
siteid_list = df_train.SiteID.unique()
# %%
fold_list = os.listdir(r'F:\fluxnet\fluxnet')
# %%
situ_var = ['SiteID', 'Year', 'Month', 'PET' 'TA_F', 'P_F(mm/mon)', 'VAP(hPa)', 'NETRAD', 'WS_F', 'PA_F(Pa)']
helper = pd.DataFrame(columns=situ_var)

for fold in fold_list:
    siteid = fold[4:10]
    if siteid in siteid_list:
        file = fold[:-13] + 'MM_' + fold[-13:] + '.csv'
        path = os.path.join(r'F:\fluxnet\fluxnet', fold, file)

        df_site = pd.read_csv(path)
        df_site.replace(-9999, np.nan, inplace=True)

        df_site['SiteID'] = siteid
        df_site['Year'] = df_site['TIMESTAMP'] // 100
        df_site['Month'] = df_site['TIMESTAMP'] % 100

        df_site['days'] = pd.PeriodIndex(year=df_site['Year'], month=df_site['Month'], freq='M').days_in_month

        df_site['P_F(mm/mon)'] = df_site['P_F'] * df_site['days']
        df_site['VAP(hPa)'] = (
            6.108 * np.exp(17.27 * df_site['TA_F'] / (df_site['TA_F'] + 237.3)) - df_site['VPD_F']
        )

        df_site['PA_F(Pa)'] = df_site['PA_F'] * 1000

        df_site['PET'] = df_site.apply(
            lambda x: PM_ET0(x['NETRAD'], x['G_F_MDS'], x['TA_F'], x['WS_F'], x['VPD_F'], x['PA_F']), axis=1
        )

        helper = pd.concat([helper, df_site[situ_var]], ignore_index=True)

        print(fold[4:10])


df_train = df_train.merge(helper, how='left', on=['SiteID', 'Year', 'Month'])
df_train.dropna(axis=0, inplace=True)

df_train.to_csv(r'D:\ET\TrainingDataProcess\TrainingList_insitu_as_input.csv')
# %%
# Split training & testing sets, train the model
# %%
df_grid = pd.read_csv(r'D:\ET\MLResults\Deepf_test_results\Deepf_test_CRO_base.csv', index_col=0)
df_situ = pd.read_csv(r'D:\ET\MLResults\Deepf_test_results\Deepf_test_CRO_insitu.csv', index_col=0)
# %%
df_grid['days'] = pd.PeriodIndex(year=df_grid['Year'], month=df_grid['Month'], freq='M').days_in_month
df_situ['days'] = pd.PeriodIndex(year=df_situ['Year'], month=df_situ['Month'], freq='M').days_in_month

df_grid['ET_true (mm/mon)'] = df_grid['EstimateLE'] * 0.035 * df_grid['days']
df_grid['ET_pred (mm/mon)'] = df_grid['predict (W/m2)'] * 0.035 * df_grid['days']

df_situ['ET_true (mm/mon)'] = df_situ['EstimateLE'] * 0.035 * df_situ['days']
df_situ['ET_pred (mm/mon)'] = df_situ['predict (W/m2)'] * 0.035 * df_situ['days']
# %%
fig, ax = plt.subplots(figsize=(4, 4), dpi=400)

test_plot(
    x=df_grid['ET_pred (mm/mon)'],
    y=df_grid['ET_true (mm/mon)'],
    color='gray',
    xlabel='ET$_{predict}$ (mm/month)$',
    ylabel='ET$_{FLUXNET}$ (mm/month)$',
    xtext=0.05,
    ytext=0.65,
    label='Source 1',
    ax=ax,
)

test_plot(
    x=df_situ['ET_pred (mm/mon)'],
    y=df_situ['ET_true (mm/mon)'],
    color='#5AA897',
    xlabel='ET$_{predict}$ (mm/month)',
    ylabel='ET$_{FLUXNET}$ (mm/month)',
    xtext=0.05,
    ytext=0.43,
    label='Source 2',
    ax=ax,
)

fig.savefig(r'D:\ET\Figures\datasource_uncertainty_insitu.png', bbox_inches='tight', dpi=400)
# %%
