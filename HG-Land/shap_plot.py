# %%
import numpy as np
import pandas as pd
import proplot as pplt

# %%
df = pd.read_excel(r"D:\ET\MLResults\shap_specific_sites.xlsx")

df_humid = (
    df[df["climate type"] == "Humid"].sort_values(by="SiteID").reset_index(drop=True)
)
df_dry = (
    df[df["climate type"] != "Humid"].sort_values(by="SiteID").reset_index(drop=True)
)
# %%
for df in [df_dry, df_humid]:
    fig, axs = pplt.subplots(
        ncols=2,
        nrows=5,
        figsize=(6, 8.5),
        wspace=4,
        hspace=1.5,
        sharex=1,
        sharey=1,
        span=True,
    )

    for i in range(len(df.index)):
        p1 = axs[i].barh(
            df.iloc[i, 7:].sort_values(ascending=True),
            width=0.6,
            color="sea",
            alpha=0.5,
        )

        axs[i].bar_label(p1, label_type="edge")
        axs[i].spines[["top", "right"]].set_visible(False)
        axs[i].format(
            abc="(a)",
            abcloc="ul",
            titleloc="lr",
            title=f"{df.iloc[i,0]}\n{df.iloc[i,3]}\n{df.iloc[i,5]}",
            xlim=(0, 1.18 * df.iloc[i, 7:].max()),
        )

    axs.format(xlabel="mean(|SHAP value|)")

    fig.savefig(
        f"D:\\ET\\Figures\\\\{df.iloc[i,5]}.tiff",
        bbox_inches="tight",
        dpi=500,
    )
