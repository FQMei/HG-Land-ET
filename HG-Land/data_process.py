# %%
import os
import pandas as pd
import xarray as xr

# %%
IGBP_list = ['CRO', 'CSH', 'DBF', 'EBF', 'ENF', 'GRA', 'MF', 'OSH', 'SAV', 'WET', 'WSA']
df_LE = pd.read_csv(r'D:\ET\TrainingDataProcess\TrainingList_Flux.csv', index_col=0)

for IGBP in IGBP_list:
    df_select = df_LE[df_LE.IGBP == IGBP]
    df_remain = df_LE[df_LE.IGBP != IGBP]

    if IGBP != 'ENF':
        n = 2195 - len(df_select)
        df_sampled = df_remain.sample(n=n)
        df_train = df_remain[~df_remain.index.isin(df_sampled.index)]
        df_test = pd.concat([df_select, df_sampled], ignore_index=True)

    else:
        df_test = df_select
        df_train = df_remain

    df_test.to_csv(f'D:\\ET\\TrainingDataProcess\\train_test_sets\\Testset_{IGBP}.csv', index=False)
    df_train.to_csv(f'D:\\ET\\TrainingDataProcess\\train_test_sets\\Trainset_{IGBP}.csv', index=False)

# %%
ncfile_dir = r'D:\ET\DataCollect\GlobalMonthlyVars'
feature_orig_dir = r'D:\ET\TrainingDataProcess\global_inputs_original'
feature_intp_dir = r'D:\ET\TrainingDataProcess\global_inputs_interp'
helper_path = r'D:\ET\TrainingDataProcess\lat_lon.csv'

features = ['tmp', 'pre', 'vap', 'pet', 'wet', 'frs', 'rad', 'wnd', 'tsk', 'prs', 'fAPAR', 'LAI']

df_helper = pd.read_csv(helper_path)

for mon in range(444):
    df_left = df_helper.copy()

    for i in features:
        file = i + '_1982_2018.nc'
        sample_path = os.path.join(ncfile_dir, file)

        ds = xr.open_dataset(sample_path)

        df = ds[i].isel(time=mon).to_dataframe()
        df_left = df_left.merge(df, how='left', on=['lat', 'lon'])

    df_left.to_csv(feature_orig_dir + f'\\{mon}.csv')

    print(f'{mon} saved')

    del df_left
# %%
for mon in range(444):
    file_path = feature_orig_dir + f'\\{mon}.csv'
    df = pd.read_csv(file_path)

    for i in features:
        df[i].mask(df[i].isna(), df[i + '_mon'], axis='index', inplace=True)

    df.to_csv(feature_intp_dir + f'\\{mon}.csv', index=False)
# %%
