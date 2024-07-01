#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 18:00:21 2024

@author: anna
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import pickle
import faiss
import numpy as np
from adjustText import adjust_text
from seaborn import scatterplot


# data = pd.read_csv("Metrics Full Results.csv")

# data = data.drop(columns = data.columns[-3])

metrics = pd.read_csv("VM_combined_metrics_2.csv", index_col=0)
glue = pd.read_csv("VM_combined_glue_2.csv", index_col = 0)

metrics = metrics.reset_index(names = "model")
metrics["alpha"] = metrics["model"].str.replace("bert-small_", "").str[:-2].astype(float)
metrics = metrics.set_index("model")

glue["new_average"] = glue.drop(['wnli', 'average'], axis = 1).mean(axis = 1)

df = pd.concat([metrics, glue["new_average"]], axis = 1)

columns = ["p-error",
"p-centroids",
"p-point_dist",
"p-patchiness",
"p-iqr",
"a-error",
"a-centroids",
"a-point_dist",
"a-patchiness",
"a-iqr",
"EEE",
"VRM",
"IsoScore",
"p-under_mu",
"p-under_mu2",
"p-skew",
"a-under_mu",
"a-under_mu2",
"a-skew",
           "p-count_var",
           "p-count_kl",
           "a-count_var",
           "a-count_kl",
           "alpha",
           "new_average"]

df.columns = columns

df2 = df.copy()

#reorder and filter for viz
cols = [
#             "p-patchiness",
# "p-count_var",
# "p-count_kl",
#     "p-skew",
#         "p-error",
#         "p-iqr",
#   "p-centroids",
#   "p-point_dist",

# "a-patchiness",
# "a-count_var",
# "a-count_kl",
# "a-skew",
# "a-error",
# "a-iqr",
# "a-centroids",
# "a-point_dist",
"EEE",
"VRM",
"IsoScore",
            "alpha",
            "new_average"]

df = df[cols]

#%%
#explore correlation
#pearson
correlation = df.drop(["alpha"], axis = 1).corr()#.drop("new_average")
correlation = correlation.rename(columns = q_short_names, index = q_short_names)
correlation = correlation.drop(["AQ % Reconstruction Below Mean",
                                "AQ % Reconstruction Below Mean/2",
                                "PQ % Reconstruction Below Mean",
                                "PQ % Reconstruction Below Mean/2"]).drop(["AQ % Reconstruction Below Mean",
                                                                "AQ % Reconstruction Below Mean/2",
                                                                "PQ % Reconstruction Below Mean",
                                                                "PQ % Reconstruction Below Mean/2"],axis = 1)

order = ["EEE",
                                           "VRM",
                                           "IS",
                                           "PQ-RS",
                                           "PQ-RI",
                                           "PQ-RE",
                                           "PQ-PP",
                                           "PQ-PC_var",
                                            "PQ-PC_kl",
                                            "PQ-CD_var",
                                            "PQ-PD_EEE",
                                            "AQ-RI",
                                            "AQ-RS",
                                            "AQ-RE",
                                            "AQ-PP",
                                            "AQ-PC_var",
                                            "AQ-PC_kl",
                                            "AQ-CD_var",
                                            "AQ-PD_EEE"]   

correlation = correlation.reindex(index = order)
correlation   = correlation[order]                                       

#plt.matshow(correlation)

plt.figure(figsize=(7,7))
sns.heatmap(correlation,#[["new_average"]], 
            xticklabels=correlation.index.values,
            yticklabels=correlation.index.values,
            annot=False,
            cbar=True)
plt.title("Pearson Correlation for All Metrics")
plt.savefig("full_correlation_matrix.pdf", format = "pdf")

correlation["new_average"]

#spearman

# correlation2 = data.corr(method = "spearman")

# plt.matshow(correlation2)

# sns.heatmap(correlation2, 
#             xticklabels=correlation2.columns.values,
#             yticklabels=correlation2.columns.values)

# correlation2["Glue Performance"]

roberta_corr = roberta_metrics.corr()
roberta_corr["average"]
#%%
#explore regression/anova

# reg = LinearRegression().fit(X = data[data.columns[2:-1]], y = data[data.columns[-1]])


#%%
#visualize relationships

plt.scatter(x =metrics["a-centroids"], y = glue["new_average"], c = metrics["alpha"], label = metrics["alpha"])
plt.scatter(x =alt_models.drop(["untrained"])["a-centroids"], y = alt_models.drop(["untrained"])["glue_average"], c = "red")
plt.colorbar().set_label("Noise Alpha", rotation=270)
plt.xlabel("Variance")
plt.ylabel("GLUE")
plt.suptitle("centroid dist vs. Average GLUE Score")
#plt.title("Additive Quantizer")
for idx,row in alt_models.drop(["untrained"]).iterrows():
    plt.annotate(row['model'], (row['a-centroids'], row['glue_average']))

#%%
# load alternative model results
alt_models = pd.read_csv("Metrics Full Results.csv")
alt_models[["Glue Performance"]] = alt_models[["Glue Performance"]]/100

columns = ["paper",
            "model",
            "p-error",
            "p-count_var",
            "p-count_kl",
            "p-centroids",
            "p-point_dist",
            "p-patchiness",
            "a-error",
            "a-count_var",
            "a-count_kl",
            "a-centroids",
            "a-point_dist",
            "a-patchiness",
            "EEE",
            "VRM",
            "Iso_string",
            "IsoScore",
            "glue_average"]

alt_models.columns = columns

model_index = ["roberta-base",
                "asci",
                "fc",
                "rand",
                "roberta_sinha",
                "shuffle_sent",
                "shuffle_corp",
                "bert-base",
                "germ",
                "chin",
                "untrained",
                "shuffle_index"]

alt_models["model_index"] = model_index
alt_models = alt_models.set_index("model_index")
#%%
#visualize all metric v glue
pretty_names = {
           "p-error":"Reconstruction Error",
           "p-count_var": "Point Count Var.",
           "p-count_kl": "Point Count KL-Div.",
           "p-centroids":"Cent. Distance Var.",
           "p-point_dist": "Point Dist. EEE",
           "p-patchiness": "Point Patchiness",
           "p-iqr": "Reconstruction IQR",
           "p-under_mu": "PQ % Reconstruction Below Mean",
           "p-under_mu2": "PQ % Reconstruction Below Mean/2",
           "p-skew": "Reconstruction Skew",
           "a-error": "Reconstruction Error",
           "a-count_var": "Point Count Var.",
           "a-count_kl": "Point Count KL-Divergence",
           "a-centroids": "Cent. Distance Var.",
           "a-point_dist": "Point Dist. EEE",
           "a-patchiness": "Point Patchiness",
           "a-iqr": "Reconstruction IQR",
           "a-under_mu": "AQ % Reconstruction Below Mean",
           "a-under_mu2": "AQ % Reconstruction Below Mean/2",
           "a-skew": "Reconstruction Skew",
           "EEE":"EEE",
           "VRM":"VRM",
           "IsoScore":"IsoScore",}

short_names = {
           "p-error":"RE",
           "p-count_var": "PC_var",
           "p-count_kl": "PC_kl",
           "p-centroids":"CD_var",
           "p-point_dist": "PD_EEE",
           "p-patchiness": "PP",
           "p-iqr": "RI",
           "p-under_mu": "PQ % Reconstruction Below Mean",
           "p-under_mu2": "PQ % Reconstruction Below Mean/2",
           "p-skew": "RS",
           "a-error": "RE",
           "a-count_var": "PC_var",
           "a-count_kl": "PC_kl",
           "a-centroids": "CD_var",
           "a-point_dist": "PD_EEE",
           "a-patchiness": "PP",
           "a-iqr": "RI",
           "a-under_mu": "AQ % Reconstruction Below Mean",
           "a-under_mu2": "AQ % Reconstruction Below Mean/2",
           "a-skew": "RS",
           "EEE":"",
           "VRM":"",
           "IsoScore":"IS",}

q_short_names = {
           "p-error":"PQ-RE",
           "p-count_var": "PQ-PC_var",
           "p-count_kl": "PQ-PC_kl",
           "p-centroids":"PQ-CD_var",
           "p-point_dist": "PQ-PD_EEE",
           "p-patchiness": "PQ-PP",
           "p-iqr": "PQ-RI",
           "p-under_mu": "PQ % Reconstruction Below Mean",
           "p-under_mu2": "PQ % Reconstruction Below Mean/2",
           "p-skew": "PQ-RS",
           "a-error": "AQ-RE",
           "a-count_var": "AQ-PC_var",
           "a-count_kl": "AQ-PC_kl",
           "a-centroids": "AQ-CD_var",
           "a-point_dist": "AQ-PD_EEE",
           "a-patchiness": "AQ-PP",
           "a-iqr": "AQ-RI",
           "a-under_mu": "AQ % Reconstruction Below Mean",
           "a-under_mu2": "AQ % Reconstruction Below Mean/2",
           "a-skew": "AQ-RS",
           "EEE":"EEE",
           "VRM":"VRM",
           "IsoScore":"IS",}

axes_lookup = [(0,0),(0,1),(0,2),(0,3),(0,4),
               (1,0),(1,1),(1,2),(1,3),(1,4),
               (2,0),(2,1),(2,2),(2,3),(2,4)
               ]

f, axes = plt.subplots(3,5, figsize = (20,15))
f.tight_layout()
plt.subplots_adjust(hspace = 0.25)   
#f.colorbar().set_label("Noise Alpha", rotation=270)
# axes[1,0].set_title('Shell', fontsize = 'xx-large')        
# sns.scatterplot(x = circ[2].T[0], y = circ[2].T[1], ax = axes[1,0])

for idx,m in enumerate(list(df.columns)):
    if m != "alpha" and m != "new_average":
        
        row,col = axes_lookup[idx]
        axes[row,col].scatter(x =df[m], y = df["new_average"], c = df["alpha"], label = metrics["alpha"])
        axes[row,col].scatter(x =alt_metrics[m], y = alt_metrics['glue_average'], c = "red")
        axes[row,col].scatter(x = base_metrics[m], y = base_metrics['average'], c = "orange")
        axes[row,col].scatter(x = roberta_metrics[m], y = roberta_metrics['average'], c = "blue")
        axes[row,col].set_xlabel(pretty_names[m])
        axes[row,col].set_ylabel("GLUE")
        axes[row,col].set_title(pretty_names[m])
        for idx,r in alt_metrics.iterrows():
            axes[row,col].annotate(r['model'], (r[m], r['glue_average']))
        # for idx,r in base_metrics.iterrows():
        #     axes[row,col].annotate(r['alpha'], (r[m], r['average']))
        # for idx,r in roberta_metrics.iterrows():
        #     axes[row,col].annotate(r['alpha'], (r[m], r['average']))
plt.savefig("all_correlations.pdf", format = "pdf")
plt.show()

#%%
#visualize bert-small, bert-base, and roberta-base w/ lines vs. alt models



axes_lookup = [(0,0),(0,1),(0,2),#(0,3),(0,4),(0,5),
               (1,0),(1,1),(1,2),#(1,3),#(1,4),#(1,5),
               (2,0),(2,1),(2,2),#(2,3),#(2,4),#(2,5),
               (3,0),(3,1),(3,2)]#,(3,3)]#,(3,4)]#,(3,5)]

# select_metrics = df.loc[df["alpha"].isin([1,3,5,10,18, 70])]
select_metrics = df.groupby("alpha").median()
select_metrics = select_metrics.sort_values(["alpha"])
select_metrics["alpha"] = select_metrics.index

base_metrics = base_metrics.sort_values(["alpha"])
roberta_metrics = roberta_metrics.groupby("alpha").median()

bert_papers = alt_metrics[alt_metrics["model"].isin(["chin", "germ", "untrained_w_emb", "shuffle_index"])]
roberta_papers = alt_metrics[alt_metrics["model"].isin(["shuffle_sent", "shuffle_corp", "roberta_sinha", "mlm", "asci", "rand", "fc"])]
bert_small_metrics = alt_metrics[alt_metrics["model"] == "bert-small"]
highlight_papers = alt_metrics[alt_metrics["model"].isin(["untrained", "untrained_w_emb", "rand"])]

# plt.plot(select_metrics[m], select_metrics["new_average"], c = "blue", label = "bert-small")
# plt.scatter(x =alt_metrics[m], y = alt_metrics['glue_average'], c = "red")
# plt.plot(base_metrics[m], base_metrics['average'], c = "orange", label = "bert-base")

# f, axes = plt.subplots(1,4, figsize = (14,4))
f, axes = plt.subplots(1,3, figsize = (10,3.3))
# f, axes = plt.subplots(2,2, figsize = (7,7))
#f, axes = plt.subplots(2,3, figsize = (10,7))
# f, axes = plt.subplots(3,3, figsize = (10,10))
#f, axes = plt.subplots(3,5, figsize = (20,20))
# f, axes = plt.subplots(2,5, figsize = (20,10))
f.tight_layout()
plt.subplots_adjust(hspace = 0.2) #wspace to control vertical space
# axes[-1].axis('off') 
# axes[-1,-1].axis('off') 
# axes[-1,-2].axis('off')  
# axes[-1,-3].axis('off')  
# axes[-1,-4].axis('off') 
#f.colorbar().set_label("Noise Alpha", rotation=270)
# axes[1,0].set_title('Shell', fontsize = 'xx-large')        
# sns.scatterplot(x = circ[2].T[0], y = circ[2].T[1], ax = axes[1,0])


# for idx,m in enumerate(list(df.columns)):
# # # for idx, m in enumerate(["p-error","p-iqr", "p-under_mu", "p-under_mu2", "p-skew",
# # #                          "a-error","a-iqr", "a-under_mu", "a-under_mu2", "a-skew"]):
#     if m != "alpha" and m != "new_average":
#         row,col = axes_lookup[idx]
#         axes[row,col].scatter(select_metrics[m], select_metrics["new_average"], c = "lightblue", label = "bert-small")
#         axes[row,col].scatter(x =bert_papers[m], y = bert_papers['glue_average'], c = "orange", edgecolor = "black")
#         axes[row,col].scatter(x =roberta_papers[m], y = roberta_papers['glue_average'], c = "limegreen", edgecolor = "black")
#         axes[row,col].scatter(x =bert_small_metrics[m], y = bert_small_metrics['glue_average'], c = "lightblue", edgecolor = "black")
#         axes[row,col].scatter(base_metrics[m], base_metrics['average'], c = "orange", label = "bert-base")
#         axes[row,col].scatter(roberta_metrics[m], roberta_metrics['average'], c = "limegreen", label = "roberta-base")
        
#         axes[row,col].set_title(pretty_names[m], fontsize="xx-large")
#         #highlight untrained & rand models
#         #axes[row,col].scatter(x =highlight_papers[m], y = highlight_papers['glue_average'], c = "red", s=100)
#         #highlight duplicated model run
#         #axes[row,col].scatter(base_metrics.at["bert-base-uncased_5_2",m], base_metrics.at["bert-base-uncased_5_2",'average'], c = "red")
#         #axes[row,col].scatter(base_metrics.at["bert-base-uncased_5_1",m], base_metrics.at["bert-base-uncased_5_1",'average'], c = "red")
#         text = [axes[row,col].text(r[m], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt_metrics[alt_metrics["model"] != "untrained"].iterrows()]
#         adjust_text(text, ax = axes[row,col],arrowprops=dict(arrowstyle='->', color='black'))
#         if m == "EEE":# or m == "p-patchiness" or m == "p-centroids":
#             axes[row,col].set_ylabel("GLUE Average", fontsize = "large")
#         if m == "IsoScore":
#             axes[row,col].legend()
#             #axes[row,col].legend(bbox_to_anchor=(1.2,1), fontsize = "xx-large")

for idx,m in enumerate(list(df.columns)):
# for idx, m in enumerate(["a-skew", "a-centroids","a-point_dist", "EEE"]):
    if m != "alpha" and m != "new_average":
        row,col = axes_lookup[idx]
        axes[col].scatter(select_metrics[m], select_metrics["new_average"], c = "lightblue", label = "bert-small")
        axes[col].scatter(x =bert_papers[m], y = bert_papers['glue_average'], c = "orange", edgecolor = "black")
        axes[col].scatter(x =roberta_papers[m], y = roberta_papers['glue_average'], c = "limegreen", edgecolor = "black")
        axes[col].scatter(x =bert_small_metrics[m], y = bert_small_metrics['glue_average'], c = "lightblue", edgecolor = "black")
        axes[col].scatter(base_metrics[m], base_metrics['average'], c = "orange", label = "bert-base")
        axes[col].scatter(roberta_metrics[m], roberta_metrics['average'], c = "limegreen", label = "roberta-base")
        
        axes[col].set_title(pretty_names[m], fontsize="xx-large")
        #highlight untrained & rand models
        #axes[row,col].scatter(x =highlight_papers[m], y = highlight_papers['glue_average'], c = "red", s=100)
        #highlight duplicated model run
        #axes[row,col].scatter(base_metrics.at["bert-base-uncased_5_2",m], base_metrics.at["bert-base-uncased_5_2",'average'], c = "red")
        #axes[row,col].scatter(base_metrics.at["bert-base-uncased_5_1",m], base_metrics.at["bert-base-uncased_5_1",'average'], c = "red")
        text = [axes[col].text(r[m], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt_metrics[alt_metrics["model"] != "untrained"].iterrows()]
        adjust_text(text, ax = axes[col],arrowprops=dict(arrowstyle='->', color='black'))
        if m == "EEE":#"a-skew": #or m == "a-patchiness": #or m == "a-centroids":
            axes[col].set_ylabel("GLUE Average", fontsize = "large")
        if m == "VRM":
            axes[col].legend(loc = "upper center",bbox_to_anchor=(0.4,-0.05), fontsize = "x-large",fancybox = False, shadow = False, ncol = 3)


plt.savefig("spread_metrics_median.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

#%%
#plot patchiness alone
m = "a-patchiness"
plt.figure(figsize = (4,4))
plt.scatter(select_metrics[m], select_metrics["new_average"], c = "lightblue", label = "bert-small")
plt.scatter(base_metrics[m], base_metrics['average'], c = "orange", label = "bert-base")
plt.scatter(roberta_metrics[m], roberta_metrics['average'], c = "limegreen", label = "roberta-base")

# plt.scatter(x =bert_papers[m], y = bert_papers['glue_average'], c = "orange", edgecolor = "black")
# plt.scatter(x =roberta_papers[m], y = roberta_papers['glue_average'], c = "limegreen", edgecolor = "black")
# plt.scatter(x =bert_small_metrics[m], y = bert_small_metrics['glue_average'], c = "lightblue", edgecolor = "black")

plt.suptitle("Point Patchiness", fontsize="x-large")
plt.title("PP")
#highlight untrained & rand models
#axes[row,col].scatter(x =highlight_papers[m], y = highlight_papers['glue_average'], c = "red", s=100)
#highlight duplicated model run
#axes[row,col].scatter(base_metrics.at["bert-base-uncased_5_2",m], base_metrics.at["bert-base-uncased_5_2",'average'], c = "red")
#axes[row,col].scatter(base_metrics.at["bert-base-uncased_5_1",m], base_metrics.at["bert-base-uncased_5_1",'average'], c = "red")
# text = [plt.text(r[m], r['glue_average'], pretty_models[r['model']], fontsize = "large") for idx,r in alt_metrics[alt_metrics["model"] != "untrained"].iterrows()]
# adjust_text(text,arrowprops=dict(arrowstyle='->', color='black'))
plt.ylabel("GLUE Average", fontsize = "large")
plt.legend()
plt.savefig("aq_patchiness.pdf", format = "pdf", bbox_inches = "tight")
plt.show()


f, axes = plt.subplot_mosaic('AB', constrained_layout = True, figsize = (8,4))
ax1 = axes["A"]
ax2 = axes["B"]

small_select = small.drop(["source","model", "type"], axis = 1).groupby("alpha").median()
roberta_select = roberta.drop(["source", "model", "type"],axis = 1).groupby("alpha").median()

axes["A"].scatter(bert["a-patchiness"], bert["glue_average"],c = "orange", label = "bert-base")
axes["A"].scatter(small_select["a-patchiness"], small_select["glue_average"], c = "lightblue", label = "bert-small")
axes["A"].scatter(roberta_select["a-patchiness"], roberta_select["glue_average"], c = "limegreen", label = "roberta")
# ax2.scatter(bert_alt["a-patchiness_m"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
# ax2.scatter(roberta_alt["a-patchiness_m"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
# ax2.scatter(small_alt["a-patchiness_m"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
axes["A"].plot([min(linreg["glue_average"]), max(linreg["glue_average"])], [min(linreg["glue_average"]), max(linreg["glue_average"])], c = "black")
# text = [ax2.text(r['a-patchiness_m'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
# adjust_text(text, ax = ax2,arrowprops=dict(arrowstyle='->', color='black'))
axes["A"].set_title("Point Patchiness", fontsize = "xx-large")
# axes["A"].set_title("AQ-PP")
axes["A"].set_ylabel("GLUE Average", fontsize = "x-large")
axes["A"].set_xlabel("Predicted GLUE Average", fontsize = "x-large")

axes["B"].scatter(bert["a-patchiness_m"], bert["glue_average"],c = "orange", label = "bert-base")
axes["B"].scatter(small_select["a-patchiness_m"], small_select["glue_average"], c = "lightblue", label = "bert-small")
axes["B"].scatter(roberta_select["a-patchiness_m"], roberta_select["glue_average"], c = "limegreen", label = "roberta")
# ax2.scatter(bert_alt["a-patchiness_m"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
# ax2.scatter(roberta_alt["a-patchiness_m"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
# ax2.scatter(small_alt["a-patchiness_m"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
axes["B"].plot([min(linreg["glue_average"]), max(linreg["glue_average"])], [min(linreg["glue_average"]), max(linreg["glue_average"])], c = "black")
# text = [ax2.text(r['a-patchiness_m'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
# adjust_text(text, ax = ax2,arrowprops=dict(arrowstyle='->', color='black'))
axes["B"].set_title("+Architecture", fontsize = "xx-large")
# axes["B"].set_title("AQ-PP")
axes["B"].set_ylabel("GLUE Average", fontsize = "x-large")
axes["B"].set_xlabel("Predicted GLUE Average", fontsize = "x-large")
axes["B"].legend(loc = "lower right")
plt.savefig("aq_patchiness_linreg.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

plt.figure(figsize = (4,4))
plt.scatter(bert["a-patchiness_m"], bert["glue_average"],c = "orange", label = "bert-base")
plt.scatter(small_select["a-patchiness_m"], small_select["glue_average"], c = "lightblue", label = "bert-small")
plt.scatter(roberta_select["a-patchiness_m"], roberta_select["glue_average"], c = "limegreen", label = "roberta")
plt.scatter(bert_alt["a-patchiness_m"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
plt.scatter(roberta_alt["a-patchiness_m"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
plt.scatter(small_alt["a-patchiness_m"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
plt.plot([min(linreg["glue_average"]), max(linreg["glue_average"])], [min(linreg["glue_average"]), max(linreg["glue_average"])], c = "black")
text = [plt.text(r['a-patchiness_m'], r['glue_average'], pretty_models[r['model']], fontsize = "large") for idx,r in alt.iterrows()]
adjust_text(text, arrowprops=dict(arrowstyle='->', color='black'))
plt.suptitle("Point Patchiness +Architecture", fontsize = "x-large")
plt.title("Non-standard Model Fit")
plt.ylabel("GLUE Average")
plt.xlabel("Predicted GLUE Average")
plt.legend(loc = "lower right")
plt.savefig("aq_patchiness_linreg_alt.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

#%%
# lm_vars = sm.add_constant(metrics[["p-error",
#                    "p-point_dist",
#                    "p-count_var",
#                    "a-error",
#                    "a-point_dist",
#                    "a-centroids",
#                    "a-count_var",
#                    "EEE",
#                    "IsoScore"]])

# lm = sm.OLS(glue["new_average"], lm_vars).fit()
# lm.summary()
# sm.stats.anova_lm(lm, typ=2)

# anova = ols("new_average ~ p_error+p_point_dist+p_count_var+a_error+a_point_dist+a_centroids+a_count_var+EEE+IsoScore", data = df).fit()
# print(sm.stats.anova_lm(anova, typ=2))

#%%
# visualizations for varying alpha
# can this explain zig-zag metric results

from sklearn.decomposition import PCA

start_int = "/media/anna/Samsung_T5/manifolds/VM_results/"
start_float = "/media/anna/Samsung_T5/manifolds/float_runs/metric_results/prajjwal1/"
#alpha_list = list(set(df["alpha"]))
alpha_list = [0.5,1.5,2.5,5,10,18,35,70]
VM_lookup = {1:"VM_1",2:"VM_1",3:"VM_1",4:"VM_1",
             5:"VM_2",6:"VM_2",7:"VM_2",10:"VM_2",
             8:"VM_3",13:"VM_3",20:"VM_3",30:"VM_3",
             9:"VM_4",15:"VM_4",40:"VM_4",
             11:"VM_5",17:"VM_5",50:"VM_5",
             12:"VM_6",23:"VM_6",60:"VM_6",
             14:"VM_7",25:"VM_7",70:"VM_7",
             16:"VM_8",27:"VM_8",80:"VM_8",
             18:"VM_9",33:"VM_9",90:"VM_9",
             19:"VM_10",35:"VM_10",100:"VM_10"}

plt.plot()
plt.title("point count histograms by alpha")
plt.xlabel("cluster")
plt.ylabel("assigned points")

hist_list = []
for idx, alpha in enumerate(alpha_list):
    print("building quantizer", str(idx))
    models = df.index[df['alpha']==alpha].to_list()
    if alpha % 1 == 0:
        VM = VM_lookup[alpha]
        m = models[0]
        f = start_int+VM+"/test/metric_results/prajjwal1/"+m+"/sample_space.pkl"
        ss = pickle.load(open(f, "rb"))
        
        #viz histogram of quantized point counts
        d = len(ss[0])
        aq = faiss.LocalSearchQuantizer(d,4,8)
        aq.train(np.array(ss))
        
        codes = aq.compute_codes(np.array(ss)).T
        all_hist = []
        for i in range(4):
            h = np.histogram(codes, bins = 256)[0]
            all_hist.extend(h)
            
        plt.hist(all_hist, bins = 30, histtype = "step", label= alpha)
        
        #save histogram data for viz finetuning
        hist_list.append((alpha, all_hist))
        
        #viz magnitude of vectors
        #use VRM as line color
        # v = df.loc[m]["VRM"]
        # c = ("blue", v)
        # n = np.linalg.norm(ss, axis = 1)
        # plt.hist(n, bins = 30, label = f"A:{alpha}/V:{np.round(v,2)}", histtype = "step", color = c)
        
        # viz cum sum of eigenvalues
        # EEE(ss, plot = True)
        
        # viz 1st two pc for clusteriness
        # pca = PCA(n_components = 2).fit(ss)
        # tx = pca.transform(ss)
        # fig = plt.figure(figsize=(4, 3))
        # plt.scatter(tx[:,0], tx[:,1], s=1, alpha=0.3, marker='.')
        # plt.title(m+"_"+str(df.loc[m]["EEE"]))
        # fig.show()
        # plt.savefig(m+".png", format = 'png')
        
        
    else:
        m = models[0]
        f = start_float+m+"/sample_space.pkl"
        ss = pickle.load(open(f, "rb"))
        
        #viz histogram of quantized point counts
        d = len(ss[0])
        aq = faiss.LocalSearchQuantizer(d,4,8)
        aq.train(np.array(ss))
        
        codes = aq.compute_codes(np.array(ss)).T
        all_hist = []
        for i in range(4):
            h = np.histogram(codes, bins = 256)[0]
            all_hist.extend(h)
            
        plt.hist(all_hist, bins = 30, histtype = "step", label = alpha)
        
        #save histogram data for viz finetuning
        hist_list.append((alpha, all_hist))
        
        #viz magnitude of vectors
        #use VRM as line color
        # v = df.loc[m]["VRM"]
        # c = ("blue", v)
        # n = np.linalg.norm(ss, axis = 1)
        # plt.hist(n, bins = 30, label = f"A:{alpha}/V:{np.round(v, 2)}", histtype = "step", color = c)
        
        #viz cum sum of eigenvalues
        # EEE(ss, plot = True)
        
        #viz 1st two pc for clusteriness
        # pca = PCA(n_components = 2).fit(ss)
        # tx = pca.transform(ss)
        # fig = plt.figure(figsize=(4, 3))
        # plt.scatter(tx[:,0], tx[:,1], s=1, alpha=0.3, marker='.')
        # plt.title(m+"_"+str(df.loc[m]["EEE"]))
        # fig.show()
        # plt.savefig(m+".png", format = 'png')
plt.legend()
plt.savefig("point_count_histograms.pdf", format = "pdf")            
plt.show()

#%%

plt.plot()
plt.title("point count histograms by alpha and aq-count_var")
plt.xlabel("assigned points")
plt.ylabel("number clusters")

l = np.min(df["a-count_var"])
m = np.max(df["a-count_var"]-l)

for alpha, hist in hist_list:
    if alpha in [0.5, 1.5, 2.,5, 35,70]:
        model = df.index[df['alpha']==alpha].to_list()[0]
        acv = df.loc[model]["a-count_var"]
        c = ("blue", (acv-l)/m)
        plt.hist(hist, histtype = "step", bins = 50, label = f"A:{alpha}/ACV:{round(acv,2)}", color = c)
    
plt.legend(title = "alpha")
plt.savefig("point_count_histograms.pdf", format = "pdf")
plt.show()

#%%
pretty_models = {"bert-small":"BERT-small",
                 "chin": "Chinese",
                 "germ": "German",
                 "untrained_w_emb": "Untrained",
                 "shuffle_index": "Shuffle",
                 "shuffle_sent": "Sequence",
                 "shuffle_corp": "Corpus",
                 "roberta_sinha": "Roberta",
                 "asci": "ASCII",
                 "rand":"Random",
                 "fc": "First-Char",
                 "mlm": "MLM"}


#%%
#visualize select linreg model results

linreg = pd.read_csv("linreg_data_export.csv")

bert = linreg[linreg["source"] == "perturbed_weights"][linreg["type"] == "bert-base"]
roberta = linreg[linreg["source"] == "perturbed_weights"][linreg["type"] == "roberta"]
small = linreg[linreg["source"] == "perturbed_weights"][linreg["type"] == "bert-small"]
bert_alt = linreg[linreg["source"] == "alt_models"][linreg["type"] == "bert-base"][linreg["model"] != "untrained"]
roberta_alt = linreg[linreg["source"] == "alt_models"][linreg["type"] == "roberta"]
small_alt = linreg[linreg["source"] == "alt_models"][linreg["type"] == "bert-small"]
alt = linreg[linreg["source"] == "alt_models"][linreg["model"] != "untrained"]


f, axes = plt.subplot_mosaic('AB;CD', constrained_layout = True, figsize = (10,7))
ax1 = axes["A"]
ax2 = axes["B"]
ax3 = axes["C"]
ax4 = axes["D"]

ax1.scatter(bert["a-patchiness"], bert["glue_average"],c = "orange", label = "bert-base")
ax1.scatter(small["a-patchiness"], small["glue_average"], c = "lightblue", label = "bert-small")
ax1.scatter(roberta["a-patchiness"], roberta["glue_average"], c = "limegreen", label = "roberta")
ax1.scatter(bert_alt["a-patchiness"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
ax1.scatter(roberta_alt["a-patchiness"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
ax1.scatter(small_alt["a-patchiness"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
ax1.plot([min(linreg["glue_average"]), 1], [min(linreg["glue_average"]), 1], c = "black")
text = [ax1.text(r['a-patchiness'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
adjust_text(text, ax = ax1,arrowprops=dict(arrowstyle='->', color='black'))
ax1.set_title("AQ Patchiness", fontsize = "xx-large")
ax1.set_ylabel("Glue Average", fontsize = "x-large")

ax2.scatter(bert["a-patchiness_m"], bert["glue_average"],c = "orange", label = "bert-base")
ax2.scatter(small["a-patchiness_m"], small["glue_average"], c = "lightblue", label = "bert-small")
ax2.scatter(roberta["a-patchiness_m"], roberta["glue_average"], c = "limegreen", label = "roberta")
ax2.scatter(bert_alt["a-patchiness_m"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
ax2.scatter(roberta_alt["a-patchiness_m"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
ax2.scatter(small_alt["a-patchiness_m"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
ax2.plot([min(linreg["glue_average"]), 1], [min(linreg["glue_average"]), 1], c = "black")
text = [ax2.text(r['a-patchiness_m'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
adjust_text(text, ax = ax2,arrowprops=dict(arrowstyle='->', color='black'))
ax2.set_title("AQ Patchiness w/ Model Type", fontsize = "xx-large")

ax3.scatter(bert["a-point_dist"], bert["glue_average"],c = "orange", label = "bert-base")
ax3.scatter(small["a-point_dist"], small["glue_average"], c = "lightblue", label = "bert-small")
ax3.scatter(roberta["a-point_dist"], roberta["glue_average"], c = "limegreen", label = "roberta")
ax3.scatter(bert_alt["a-point_dist"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
ax3.scatter(roberta_alt["a-point_dist"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
ax3.scatter(small_alt["a-point_dist"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
ax3.plot([min(linreg["glue_average"]), 1], [min(linreg["glue_average"]), 1], c = "black")
text = [ax3.text(r['a-point_dist'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
adjust_text(text, ax = ax3,arrowprops=dict(arrowstyle='->', color='black'))
ax3.set_title("AQ Point Distribution EEE", fontsize = "xx-large")
ax3.set_ylabel("Glue Average", fontsize = "x-large")
ax3.set_xlabel("Predicted Glue Average", fontsize = "x-large")

ax4.scatter(bert["p-point_dist"],bert["glue_average"], c = "orange", label = "bert-base")
ax4.scatter(small["p-point_dist"], small["glue_average"], c = "lightblue", label = "bert-small")
ax4.scatter(roberta["p-point_dist"], roberta["glue_average"], c = "limegreen", label = "roberta")
ax4.scatter(bert_alt["p-point_dist"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
ax4.scatter(roberta_alt["p-point_dist"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
ax4.scatter(small_alt["p-point_dist"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
ax4.plot([min(linreg["glue_average"]), 1], [min(linreg["glue_average"]), 1], c = "black")
text = [ax4.text(r['p-point_dist'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
adjust_text(text, ax = ax4,arrowprops=dict(arrowstyle='->', color='black'))
ax4.set_title("PQ Point Distribution EEE", fontsize = "xx-large")
ax4.set_xlabel("Predicted Glue Average", fontsize = "x-large")

plt.savefig("linreg_compare.pdf", format = "pdf")

#%%
#visualize relationship between alpha level and glue score

plt.scatter(df["alpha"]/100, df["new_average"], alpha = 0.5)
plt.title("Weight perturbation vs. GLUE score")
plt.xlabel("Alpha (Noise Proportion)")
plt.ylabel("Avg GLUE Score")
plt.savefig("alpha_glue.pdf", format = "pdf")


#%%
#visualize select linreg model results

linreg = pd.read_csv("linreg_data_export.csv")

bert = linreg[linreg["source"] == "perturbed_weights"][linreg["type"] == "bert-base"]
roberta = linreg[linreg["source"] == "perturbed_weights"][linreg["type"] == "roberta"]
small = linreg[linreg["source"] == "perturbed_weights"][linreg["type"] == "bert-small"]
bert_alt = linreg[linreg["source"] == "alt_models"][linreg["type"] == "bert-base"][linreg["model"] != "untrained"]
roberta_alt = linreg[linreg["source"] == "alt_models"][linreg["type"] == "roberta"]
small_alt = linreg[linreg["source"] == "alt_models"][linreg["type"] == "bert-small"]
alt = linreg[linreg["source"] == "alt_models"][linreg["model"] != "untrained"]


# f, axes = plt.subplot_mosaic('A;B;C', constrained_layout = True, figsize = (5,10))
# ax1 = axes["A"]
# ax2 = axes["B"]
# ax3 = axes["C"]
#ax4 = axes["D"]

# ax1.scatter(bert["a-patchiness"], bert["glue_average"],c = "orange", label = "bert-base")
# ax1.scatter(small["a-patchiness"], small["glue_average"], c = "lightblue", label = "bert-small")
# ax1.scatter(roberta["a-patchiness"], roberta["glue_average"], c = "limegreen", label = "roberta")
# ax1.scatter(bert_alt["a-patchiness"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
# ax1.scatter(roberta_alt["a-patchiness"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
# ax1.scatter(small_alt["a-patchiness"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
# ax1.plot([min(linreg["glue_average"]), 1], [min(linreg["glue_average"]), 1], c = "black")
# text = [ax1.text(r['a-patchiness'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
# adjust_text(text, ax = ax1,arrowprops=dict(arrowstyle='->', color='black'))
# ax1.set_title("AQ Patchiness", fontsize = "xx-large")
# ax1.set_ylabel("Glue Average", fontsize = "x-large")

# ax2.scatter(bert["a-patchiness_m"], bert["glue_average"],c = "orange", label = "bert-base")
# ax2.scatter(small["a-patchiness_m"], small["glue_average"], c = "lightblue", label = "bert-small")
# ax2.scatter(roberta["a-patchiness_m"], roberta["glue_average"], c = "limegreen", label = "roberta")
# ax2.scatter(bert_alt["a-patchiness_m"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
# ax2.scatter(roberta_alt["a-patchiness_m"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
# ax2.scatter(small_alt["a-patchiness_m"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
# ax2.plot([min(linreg["glue_average"]), 1], [min(linreg["glue_average"]), 1], c = "black")
# text = [ax2.text(r['a-patchiness_m'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
# adjust_text(text, ax = ax2,arrowprops=dict(arrowstyle='->', color='black'))
# ax2.set_ylabel("Glue Average", fontsize = "x-large")
# ax2.set_title("AQ Patchiness w/ Model Type", fontsize = "xx-large")

plt.figure(figsize = (4,4))
plt.scatter(bert["a-point_dist_m"], bert["glue_average"],c = "orange", label = "bert-base")
plt.scatter(small["a-point_dist_m"], small["glue_average"], c = "lightblue", label = "bert-small")
plt.scatter(roberta["a-point_dist_m"], roberta["glue_average"], c = "limegreen", label = "roberta")
plt.scatter(bert_alt["a-point_dist_m"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
plt.scatter(roberta_alt["a-point_dist_m"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
plt.scatter(small_alt["a-point_dist_m"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
plt.plot([min(linreg["glue_average"]), max(linreg["glue_average"])+0.1], [min(linreg["glue_average"]), max(linreg["glue_average"])+0.1], c = "black")
text = [plt.text(r['a-point_dist_m'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
adjust_text(text, arrowprops=dict(arrowstyle='->', color='black'))
plt.suptitle("Point Distribution EEE +Architecture", fontsize = "x-large")
plt.title("Non-standard Model Fit")
plt.ylabel("GLUE Average", fontsize = "large")
plt.xlabel("Predicted GLUE Average", fontsize = "large")
plt.legend(loc = "lower right")
plt.savefig("pdEEE_linreg_alt.pdf", format = "pdf", bbox_inches = "tight")
plt.show()

# ax4.scatter(bert["p-point_dist"],bert["glue_average"], c = "orange", label = "bert-base")
# ax4.scatter(small["p-point_dist"], small["glue_average"], c = "lightblue", label = "bert-small")
# ax4.scatter(roberta["p-point_dist"], roberta["glue_average"], c = "limegreen", label = "roberta")
# ax4.scatter(bert_alt["p-point_dist"], bert_alt["glue_average"], c = "orange", edgecolor = "black")
# ax4.scatter(roberta_alt["p-point_dist"], roberta_alt["glue_average"], c = "limegreen", edgecolor = "black")
# ax4.scatter(small_alt["p-point_dist"], small_alt["glue_average"], c = "lightblue", edgecolor = "black")
# ax4.plot([min(linreg["glue_average"]), 1], [min(linreg["glue_average"]), 1], c = "black")
# text = [ax4.text(r['p-point_dist'], r['glue_average'], pretty_models[r['model']], fontsize = "x-large") for idx,r in alt.iterrows()]
# adjust_text(text, ax = ax4,arrowprops=dict(arrowstyle='->', color='black'))
# ax4.set_title("PQ Point Distribution EEE", fontsize = "xx-large")
# ax4.set_xlabel("Predicted Glue Average", fontsize = "x-large")

plt.savefig("linreg_compare.pdf", format = "pdf")