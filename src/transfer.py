from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from scipy.spatial import distance
from scipy.cluster import hierarchy
import statsmodels.stats.multitest as mt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import LocallyLinearEmbedding, MDS,TSNE
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import os
import yaml
import argparse
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt 
import seaborn as sns
import sys
import pandas as pd
import numpy as np
# from cdlib import algorithms
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, silhouette_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
# import community.community_louvain as community_louvain
from collections import defaultdict
from typing import List,Dict, Union
# import community_utils as cu
import utils
import ORCurvature as nc
import plotting_utils as pu
sns.set_theme(rc={'figure.figsize':(14,14)})
from sklearn.feature_selection import SelectKBest, chi2,mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.decomposition import FastICA


def main():
	drug = "Nivo"
	tissue = "SKCM"
	expression_file = f"../data/cri/{drug}/{tissue}/expression_full.csv"
	feature_file = f"../data/cri/{drug}/{tissue}/features.csv"
	response_file = f"../data/cri/{drug}/{tissue}/response.csv"


	# de_file = f"../results/{drug}/{tissue}/trimmed/manifold/DE_genes.csv"
	de_file = f"../results/{drug}/{tissue}/manifold/DE_genes.csv"
	p_thresh = 0.05

	# res_dir = f"{res_base}/{drug}/{tissue}"
	# fig_dir = f"{fig_base}/{drug}/{tissue}"

	# os.makedirs(res_dir,exist_ok=True)
	# os.makedirs(fig_dir,exist_ok=True)

	# binarize:bool = True

	expression = pd.read_csv(expression_file)
	response = pd.read_csv(response_file)



	de = pd.read_csv(de_file,index_col = 0)
	# de = de.head(30)
	genes = list(de['Gene'].values)

	expression = expression[['Run_ID']+genes]
	expression = expression.applymap(lambda x: x if isinstance(x,str) else np.log2(x+1))
	expression = expression.merge(response, on = 'Run_ID')

	ps = []
	for gene in genes:
		temp = expression[[gene,'Response']]
		r_exp = temp[temp['Response']==1][gene].values
		nr_exp = temp[temp['Response']==0][gene].values
		u,p = mannwhitneyu(r_exp, nr_exp)
		ps.append(p)
		# print(p)

	adj_ps = mt.multipletests(ps, method = 'fdr_bh')
	# print(
	print(adj_ps[1][np.argsort(adj_ps[1])])
	# sys.exit(1)
	# print(adj_ps)
	keep_idx = np.where(adj_ps[1]<=p_thresh)
	print(keep_idx)
	# print(adj_ps[1][keep_idx])
	# print(np.argsort(adj_ps[1][keep_idx]))
	

	keep_genes = [genes[x] for x in keep_idx[0]]
	# print(np.argsort(keep_idx[0]))
	# print(keep_idx[0])
	expression.drop(columns = ['Run_ID'],inplace=True)

	data = expression.melt(id_vars=['Response'],var_name = 'Gene')
	data = data[data['Gene'].isin(keep_genes)]
	ax = sns.boxplot(data=data,y='value', x='Gene',hue= 'Response')
	ax.set(ylabel = "Log2 TPM+1 Expression", xlabel = "Gene")
	plt.xticks(rotation=60)

	plt.legend(ncol=2,frameon=False)

	# sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

	plt.tight_layout()
	# plt.show()
	plt.savefig(f"../figs/{drug}/{tissue}/de_genes.png",dpi=500)
	plt.close()
	

			


	
	

	de_file = f"../results/{drug}/{tissue}/trimmed/manifold/DE_genes.csv"
if __name__ == '__main__':
	main()