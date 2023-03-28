# %%
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from deepforest import CascadeForestRegressor
import matplotlib.pyplot as plt
from PyALE import ale
import shap
# %%
feature = ['tmp', 'pre', 'vap', 'pet', 'wet', 'frs',
           'rad', 'wnd', 'tsk', 'prs', 'fAPAR', 'LAI']
# %%
train_path = r'D:\ET_FQM\ETsubject\DataProcess\TrainingList_Flux.xlsx'

fluxnet_data = pd.read_excel(train_path)

X = fluxnet_data[feature]
Y = fluxnet_data['EstimateLE']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
# %%
model = CascadeForestRegressor()
model.load(r'D:\ET_FQM\ETsubject\MLResults\Deepf_model')
# %%
explainer = shap.Explainer(model.predict, X)
shap_values = explainer(X.sample(250))
# %%
bar_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\shap_bar.png'

clustering = shap.utils.hclust(X, Y)
shap.plots.bar(shap_values, max_display=12,
               show=False, clustering=clustering)
plt.savefig(bar_path, dpi=200, bbox_inches='tight')
plt.show()
# %%
besm_path = r'D:\ET_FQM\ETsubject\Analysis\Figures\FigureA\shap_beeswarm.png'

plt.figure(figsize=(2, 1.8))
shap.plots.beeswarm(shap_values, max_display=12,
                    color=plt.get_cmap('plasma'), show=False)
plt.savefig(besm_path, bbox_inches='tight', dpi=200)
plt.show()
# %%
xlabels = ['tmp (degree Celsius)', 'pre (mm/month)', 'vap (hPa)', 'pet (mm/day)', 'wet (days/month)', 'frs (days/month)',
           r'rad (J/$m^2$)', 'wnd (m/s)', 'tsk (K)', 'prs (Pa)', 'fAPAR', 'LAI']

ale_dir = r'D:\ET_FQM\ETsubject\Analysis\Figures'
# %%
for i in range(12):

    fig, ax = plt.subplots(figsize=(3, 2), dpi=200)

    ale_eff = ale(X=X_train, model=model, feature=[feature[i]], grid_size=50,
                  include_CI=True, C=0.95, ax=ax, fig=fig)

    ax.set_title('')
    ax.get_legend().remove()
    ax.set(xlabel=xlabels[i], ylabel='ALE of ET (W/$m^2$)')

    fig.savefig(ale_dir+'\\ale{i}.png', bbox_inches='tight', dpi=200)
    plt.show()
# %%
