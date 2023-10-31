# %%
import math
import pandas as pd
import numpy as np
import xarray as xr
import xcdat as xc
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

# %%
mrb_mask = xr.open_dataset(r"D:\ET\DataCollect\Runoff\GRDC\mrb_sel_shp\mrb_mask.nc")
mrb_id = pd.read_csv(
    r"D:\ET\DataCollect\Runoff\GRDC\mrb_sel_shp\mrb_info.csv", index_col=0
)
# %%
# CLASS
clas = xc.open_mfdataset(r"D:\ET\DataCollect\CLASS\CLASS_v1-1_200*.nc")

clas.lat.attrs["axis"] = "Y"
clas.lon.attrs["axis"] = "X"
# %%
helper = pd.DataFrame(index=clas.time)

days = clas.time.dt.daysinmonth.data

for index in mrb_id.index:
    et = (
        clas.where(mrb_mask.region == index)
        .spatial.average("hfls", axis=["X", "Y"])["hfls"]
        .compute()
    )
    df = (et * 0.035 * days).to_dataframe()

    df.rename(columns={"hfls": mrb_id.at[index, "MRBID"]}, inplace=True)
    helper = pd.merge(helper, df, left_index=True, right_index=True)

helper.to_csv(r"D:\ET\BasinCalc\wb_closure\class_mrb_et.csv")
# %%
# HG-Land
hget = xc.open_dataset(r"D:\ET\MLResults\HG-Land_ET_1982-2018_MO.nc")

hget.lat.attrs["axis"] = "Y"
hget.lon.attrs["axis"] = "X"

hget_sel = hget.sel(time=slice("2003-01", "2009-12"))
# %%
helper = pd.DataFrame(index=hget_sel.time)

days = hget_sel.time.dt.daysinmonth.data

for index in mrb_id.index:
    et = hget_sel.where(mrb_mask.region == index).spatial.average(
        "LE", axis=["X", "Y"]
    )["LE"]
    df = (et * 0.035 * days).to_dataframe()
    df.rename(columns={"LE": mrb_id.at[index, "MRBID"]}, inplace=True)
    helper = pd.merge(helper, df, left_index=True, right_index=True)

helper.to_csv(r"D:\ET\BasinCalc\wb_closure\hget_mrb_et.csv")
# %%
# GLEAM
glma = xc.open_dataset(
    r"D:\ET\DataCollect\ETProducts\Gleam\v3.6a\monthly\E_1980-2021_GLEAM_v3.6a_MO_0.5deg.nc"
)

glma_sel = glma.sel(time=slice("2003-01", "2009-12"))
# %%
helper = pd.DataFrame(index=glma_sel.time)

for index in mrb_id.index:
    et = glma_sel.where(mrb_mask.region == index).spatial.average("E", axis=["X", "Y"])[
        "E"
    ]
    df = et.to_dataframe()
    df.rename(columns={"E": mrb_id.at[index, "MRBID"]}, inplace=True)
    helper = pd.merge(helper, df, left_index=True, right_index=True)

helper.to_csv(r"D:\ET\BasinCalc\wb_closure\glma_mrb_et.csv")
# %%
# FLUXCOM
flux = xc.open_mfdataset(
    r"D:\ET\DataCollect\ETProducts\FluxCom\LE.RS_METEO.EBC-ALL.MLM-ALL.METEO-CRUNCEP_v8\*.nc"
)

flux.lat.attrs["axis"] = "Y"
flux.lon.attrs["axis"] = "X"

flux_sel = flux.sel(time=slice("2003-01", "2009-12")).compute()
# %%
helper = pd.DataFrame(index=flux_sel.time)

days = flux_sel.time.dt.daysinmonth.data

for index in mrb_id.index:
    et = flux_sel.where(mrb_mask.region == index).spatial.average(
        "LE", axis=["X", "Y"]
    )["LE"]
    df = (et * days / 2.45).to_dataframe()
    df.rename(columns={"LE": mrb_id.at[index, "MRBID"]}, inplace=True)
    helper = pd.merge(helper, df, left_index=True, right_index=True)

helper.to_csv(r"D:\ET\BasinCalc\wb_closure\flux_mrb_et.csv")
# %%


def valid_metric(y_true, y_pred):
    r = pearsonr(y_true, y_pred)[0]

    MSE = mean_squared_error(y_true, y_pred)
    RMSE = math.sqrt(MSE)

    MAE = mean_absolute_error(y_true, y_pred)

    RB = np.sum(y_pred - y_true) / np.sum(y_true) * 100

    return r, RMSE, MAE, RB


# %%
df_clas = pd.read_csv(r"D:\ET\BasinCalc\wb_closure\class_mrb_et.csv")
df_hget = pd.read_csv(r"D:\ET\BasinCalc\wb_closure\hget_mrb_et.csv")
df_flux = pd.read_csv(r"D:\ET\BasinCalc\wb_closure\flux_mrb_et.csv")
df_glma = pd.read_csv(r"D:\ET\BasinCalc\wb_closure\glma_mrb_et.csv")
# %%
df_clas["time"] = pd.to_datetime(df_clas["time"])
df_hget["time"] = pd.to_datetime(df_hget["time"])
df_flux["time"] = pd.to_datetime(df_flux["time"])
df_glma["time"] = pd.to_datetime(df_glma["time"])
# %%
df_clas_yr = df_clas.groupby(by=df_clas.time.dt.year).mean() * 12
df_hget_yr = df_hget.groupby(by=df_hget.time.dt.year).mean() * 12
df_flux_yr = df_flux.groupby(by=df_flux.time.dt.year).mean() * 12
df_glma_yr = df_glma.groupby(by=df_glma.time.dt.year).mean() * 12
# %%
df_clas_yr.to_csv(r"D:\ET\BasinCalc\wb_closure\class_mrb_et_mmyr.csv")
df_hget_yr.to_csv(r"D:\ET\BasinCalc\wb_closure\hget_mrb_et_mmyr.csv")
df_flux_yr.to_csv(r"D:\ET\BasinCalc\wb_closure\flux_mrb_et_mmyr.csv")
df_glma_yr.to_csv(r"D:\ET\BasinCalc\wb_closure\glma_mrb_et_mmyr.csv")

# %%
df_list = [df_hget, df_flux, df_glma]
filename_list = ["hget", "flux", "glma"]

df_metric = pd.DataFrame(columns=["r", "RMSE(mm/mon)", "MAE(mm/mon)", "RB(%)"])

for df, filename in zip(df_list, filename_list):
    for mrbid in df_clas.columns[1:]:
        y_true = df_clas[mrbid]
        y_pred = df[mrbid]

        result = valid_metric(y_true=y_true, y_pred=y_pred)
        df_metric.loc[mrbid] = result

    df_metric.to_csv(f"D:\\ET\\BasinCalc\\wb_closure\\{filename}_metric.csv")
