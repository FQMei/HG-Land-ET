#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# %%
import os
import numpy as np
import pandas as pd
import xarray as xr
# %%
mask_path = r'D:\ET_FQM\ETsubject\Analysis\Mask_GlobalET.nc'


def mask(ds, label='land'):
    landsea = xr.open_dataset(mask_path)['mask']
    ds.coords['mask'] = (('lat', 'lon'), landsea.values)
    if label == 'land':
        ds = ds.where(ds.mask < 1)
    elif label == 'ocean':
        ds = ds.where(ds.mask > 0)
    return ds


# %%
result_dir = f'D:\\ET_FQM\\ETsubject\\MLResults\\Deepf_result'

file_list = sorted(os.listdir(result_dir), key=lambda x: int(x[:-4]))

ET_list = [np.load(os.path.join(result_dir, file)) for file in file_list]

ET_arr = np.stack(ET_list, axis=0)
# %%
save_path = r'D:\ET_FQM\ETsubject\MLResults\HG-Land_ET_1982-2018_MO.nc'

lat = np.arange(-89.75, 90., 0.5)
lon = np.arange(-179.75, 180., 0.5)
time = pd.date_range('19820101', periods=444, freq='MS')
# %%
ds = xr.Dataset(
    data_vars={'LE': (('time', 'lat', 'lon'),
                      np.where(ET_arr > 0, ET_arr, 0))},
    
    coords={'time': time, 'lat': lat, 'lon': lon},

    attrs={'Dataset': 'High-Generalized Land Evapotranspiration dataset',
           'Temporal_Resolution': 'monthly',
           'Geospatial_Resolution': '0.5 degree grid',
           'History': 'Created on 2023-03-20T12:34:54Z'
           },
)

ds['LE'].attrs['units'] = 'W m-2'
ds['LE'].attrs['long_name'] = 'latent heat'
ds['LE'].attrs['standard_name'] = 'LE'

ds_mask = mask(ds, 'ocean')
# %%
ds_drop = ds_mask.reset_coords(names=['mask'], drop=True)
ds_drop.to_netcdf(path=save_path, mode='w', format='NETCDF4')
# %%
