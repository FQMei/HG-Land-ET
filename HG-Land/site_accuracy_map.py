# %%
import pandas as pd
import matplotlib.pyplot as plt
import proplot as pplt
import cartopy.crs as ccrs
import numpy as np

# %%
metric_dir = r"D:\ET\MLResults\Deepf_test_results"
metric_hget = pd.read_csv(metric_dir + r"\hget_metrics.csv")
metric_flux = pd.read_csv(metric_dir + r"\flux_metrics.csv")
metric_glma = pd.read_csv(metric_dir + r"\gleamet_metrics.csv")


# %%
def metric_plot(fig, ax, metric, lon, lat, vmin, vmax, title, cmap):
    m = ax.scatter(
        x=lon,
        y=lat,
        s=5,
        c=metric,
        cmap=cmap,
        colorbar=False,
        vmin=vmin,
        vmax=vmax,
        transform=ccrs.PlateCarree(),
        zorder=2,
    )
    ax.format(
        land=True,
        ocean=True,
        latlim=(-60, 90),
        landcolor="gray3",
        lonlines=60,
        latlines=30,
        labels=True,
        titleloc="l",
        title=title,
    )

    ix = ax.inset_axes([0.08, 0.13, 0.17, 0.3], transform="axes", proj="cart")
    ix.format(xlabel="", ticklabelsize=8)
    ix.hist(
        metric,
        bins=20,
        weights=np.zeros_like(metric) + 1 / len(metric),
        filled=True,
        alpha=0.3,
        c="red5",
    )

    cax = fig.add_axes([0.58, 0.25, 0.2, 0.03])
    cbar = plt.colorbar(
        m, cax=cax, orientation="horizontal", pad=0.06, extend="both", aspect=30
    )
    cbar.ax.tick_params(which="both", labelsize=8, direction="in")


# %%
df_list = [metric_hget, metric_glma, metric_flux]
file_list = ["hget", "glma", "flux"]

titles = ["(a) R", "(b) RMSE (mm/month)", "(c) MAE (mm/month)", "(d) RB (%)"]
vmins = [-1, 0, 0, -100]
vmaxs = [1, 165, 140, 100]
columns = ["r", "RMSE(mm/mon)", "MAE(mm/mon)", "RB(%)"]

for df, file_name in zip(df_list, file_list):
    for column, vmin, vmax, title in zip(columns, vmins, vmaxs, titles):
        fig, ax = pplt.subplots(figsize=(4, 2.4), proj="cyl")

        cmap = ["terrain_r" if column == "r" else "terrain"][0]

        metric_plot(
            fig,
            ax,
            metric=df[column],
            lon=df.longitude,
            lat=df.latitude,
            vmin=vmin,
            vmax=vmax,
            title=title,
            cmap=cmap,
        )

        fig.savefig(
            f"D:\\ET\\Figures\\{file_name}_{column[:3]}.png",
            bbox_inches="tight",
            dpi=400,
        )

        print(file_name)
# %%
