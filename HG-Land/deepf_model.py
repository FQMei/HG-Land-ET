# %%
import os
import pandas as pd
import numpy as np
from deepforest import CascadeForestRegressor
import matplotlib.pyplot as plt
import proplot as pplt
from PyALE import ale
import shap

# %%
feature = [
    "tmp",
    "pre",
    "vap",
    "pet",
    "wet",
    "frs",
    "rad",
    "wnd",
    "tsk",
    "prs",
    "fAPAR",
    "LAI",
]


# %%
def get_features(trainfile, testfile):
    df_train = pd.read_csv(trainfile)
    df_test = pd.read_csv(testfile)

    x_train = df_train[feature]
    y_train = df_train["EstimateLE"]

    x_test = df_test[feature]
    y_test = df_test["EstimateLE"]

    y_info = df_test[["EstimateLE", "SiteID", "Year", "Month", "IGBP"]]
    return df_train, df_test, x_train, x_test, y_train, y_test, y_info


def train_Deepforest(x_train, y_train, estm=2, trees=100):
    Deepforest = CascadeForestRegressor(
        backend="sklearn",
        n_estimators=estm,
        n_trees=trees,
        partial_mode=True,
        random_state=2,
    )

    Deepforest.fit(x_train, y_train)
    return Deepforest


def get_test_predict(model, x_test, y_info, y_test):
    y_predict = model.predict(x_test)

    df_test = pd.DataFrame(
        data={"EstimateLE": y_test, "predict (W/m2)": y_predict.ravel()}
    )

    df_test = pd.merge(df_test, y_info, right_index=True, left_index=True)
    return df_test


def get_one_result_all(sample_path, model):
    X = pd.read_csv(sample_path).loc[:, feature]
    X.fillna(value=0, inplace=True)

    result = model.predict(X)
    result = result.reshape(360, 720)
    return result


# %%
# 11 models
pft_as_test = (
    "MF"  # 'CRO', 'CSH', 'DBF', 'EBF', 'ENF', 'GRA', 'OSH', 'SAV', 'WET', 'WSA'
)

train_file = f"D:\\ET\\TrainingDataProcess\\train_test_sets\\Trainset_{pft_as_test}.csv"
test_file = f"D:\\ET\\TrainingDataProcess\\train_test_sets\\Testset_{pft_as_test}.csv"

inputs_dir = f"D:\\ET\\DataProcess\\global_inputs_interp"
output_dir = f"D:\\ET\\MLResults\\Deepf_result_ensemble\\Deepf_{pft_as_test}"

df_train, df_test, X_train, X_test, Y_train, Y_test, Y_info = get_features(
    train_file, test_file
)
dpmodel = train_Deepforest(X_train, Y_train, estm=8)
df_test = get_test_predict(model=dpmodel, x_test=X_test, y_test=Y_test, y_info=Y_info)
df_test.to_csv(f"D:\\ET\\MLResults\\Deepf_test_{pft_as_test}.csv")

for mon in range(444):
    sample_path = os.path.join(inputs_dir, f"{mon}.csv")
    result = get_one_result_all(sample_path, dpmodel)
    np.save(output_dir + f"\\{mon}.npy", result)

    print("File {} compeleted".format(mon))
# %%
# all samples for model training
trainfile = r"D:\ET\TrainingDataProcess\TrainingList_Flux.csv"

df_train = pd.read_csv(trainfile)
x_train = df_train[feature]
y_train = df_train["EstimateLE"]

Deepforest = train_Deepforest(x_train, y_train, estm=8)

for mon in range(444):
    sample_path = os.path.join(inputs_dir, f"{mon}.csv")

    result = get_one_result_all(sample_path, dpmodel)
    np.save(
        f"D:\\ET\\MLResults\\Deepf_result_ensemble\\Deepf_result\\{mon}.npy",
        result,
    )

    del result
    print("File {} compeleted".format(mon))

# Deepforest.save(f"D:\\ET\\MLResults\\Deepf_model_all_sample")
# %%
# SHAP
sites = [
    "BE-Vie",
    "BR-Sa3",
    "CA-Qfo",
    "CA-TP3",
    "CG-Tch",
    "CH-Fru",
    "CN-Cha",
    "IT-Isp",
    "IT-PT1",
    "PA-SPs",
    "AU-Cum",
    "AU-DaS",
    "AU-Gin",
    "AU-Stp",
    "AU-TTE",
    "CA-SF1",
    "ES-Amo",
    "US-Atq",
    "US-SRC",
    "US-Twt",
]

explainer = shap.Explainer(Deepforest.predict, x_train)

for site in sites:
    x = df_train[df_train["SiteID"].isin(list(site))][feature]
    shap_values = explainer(x)

    shap.plots.bar(shap_values, max_display=12, show=False)

    plt.savefig(
        f"D:\\ET\\Figures\\shap_{site}.png",
        dpi=400,
        bbox_inches="tight",
    )
    plt.close()


# %%
# ALE
xlabels = [
    "tmp (degree Celsius)",
    "pre (mm/month)",
    "vap (hPa)",
    "pet (mm/day)",
    "wet (days/month)",
    "frs (days/month)",
    r"rad (J/$m^2$)",
    "wnd (m/s)",
    "tsk (K)",
    "prs (Pa)",
    "fAPAR (-)",
    "LAI($m^2$/$m^2$)",
]

fig, axs = pplt.subplots(
    ncols=4, nrows=3, figsize=(8, 5), dpi=400, sharey="labels", sharex=False
)

axs.format(abc="(a)", abcloc="ul")

for i in range(12):
    ale.ale(
        X_train,
        dpmodel,
        [feature[i]],
        xlabels[i],
        grid_size=50,
        include_CI=True,
        C=0.95,
        ax=axs[i],
        fig=fig,
    )

    fig.savefig(f"D:\\ET\\Figures\\ale_{pft_as_test}.png", bbox_inches="tight", dpi=400)
# %%
