#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# %%
import os
import pandas as pd
import xarray as xr
# %%
features = ['tmp', 'pre', 'vap', 'pet', 'wet', 'frs',
            'rad', 'wnd', 'tsk', 'prs', 'fAPAR', 'LAI']
# %%

ncfile_dir = r'D:\ET_FQM\DataCollect\GlobalMonthlyVars'
feature_orig_dir = r'D:\ET_FQM\ETsubject\DataProcess\global_inputs_original'
helper_path = r'D:\ET_FQM\ETsubject\DataProcess\lat_lon.csv'
# %%
df_helper = pd.read_csv(helper_path)

for mon in range(444):
    df_left = df_helper.copy()

    for i in features:
        file = i+'_1982_2018.nc'
        sample_path = os.path.join(ncfile_dir, file)

        ds = xr.open_dataset(sample_path)
        df = ds[i].isel(time=mon).to_dataframe()
        df_left = df_left.merge(df, how='left', on=['lat', 'lon'])

    df_left.to_csv(feature_orig_dir+f'\{mon}.csv')

    print(f'{mon} saved')

    del df_left
# %%
feature_orig_dir = r'D:\ET_FQM\ETsubject\DataProcess\global_inputs_original'
feature_intp_dir = r'D:\ET_FQM\ETsubject\DataProcess\global_inputs_interp'
# %%
for mon in range(444):

    file_path = feature_orig_dir+f'\\{mon}.csv'
    df = pd.read_csv(file_path)

    for i in features:
        df[i].mask(df[i].isna(), df[i+'_mon'], axis='index', inplace=True)

    df.to_csv(feature_intp_dir+f'\\{mon}.csv', index=False)
# %%
