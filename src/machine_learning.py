from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, silhouette_score
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut
from scipy.stats import mannwhitneyu
import statsmodels.stats.multitest as mt
from scipy.spatial import distance
from scipy.cluster import hierarchy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import LocallyLinearEmbedding, MDS,TSNE
from sklearn.model_selection import GridSearchCV
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
# sns.set_theme()
from sklearn.feature_selection import SelectKBest, chi2,mutual_info_classif
from sklearn.metrics import silhouette_score
from sklearn.decomposition import FastICA
from collections import defaultdict

def main(config:Dict):
	drug = "Nivo"
	tissue = "SKCM"
	expression_file = f"../data/cri/{drug}/{tissue}/expression_full.csv"
	feature_file = f"../data/cri/{drug}/{tissue}/features.csv"
	response_file = f"../data/cri/{drug}/{tissue}/response.csv"
	

	# res_dir = f"{res_base}/{drug}/{tissue}"
	# fig_dir = f"{fig_base}/{drug}/{tissue}"

	# os.makedirs(res_dir,exist_ok=True)
	# os.makedirs(fig_dir,exist_ok=True)

	# binarize:bool = True

	expression = pd.read_csv(expression_file)
	response = pd.read_csv(response_file)

	seed:int = 12345
	rng = np.random.default_rng(seed)
	rstate = np.random.RandomState(seed)



	
	

	de_file = f"../results/{drug}/{tissue}/manifold/DE_genes.csv"
	# de_file = f"../results/{drug}/{tissue}/trimmed/manifold/DE_genes.csv"
	de = pd.read_csv(de_file,index_col = 0)
	# de = de.head(30)
	genes = list(de['Gene'].values)
	
	X = expression[genes]
	X = X.applymap(lambda x: np.log2(x+1))
	if drug=='Pembro':
		if tissue =='STAD':
			dex = 2
		else:
			dex = 3
	elif drug =='Nivo':
		dex = 4
	row_linkage = hierarchy.linkage(distance.pdist(X.corr(numeric_only=True)), method='average')
	res = hierarchy.fcluster(row_linkage,t=row_linkage[-dex,2], criterion='distance')
	
	X_corr = X.corr(numeric_only=True)
	
	X_corr['cluster']=res
	
	attrs = [X_corr['cluster'].values,X_corr.index]
	tuples = list(zip(*attrs))
	index = pd.MultiIndex.from_tuples(tuples, names=["Module", "Gene"])
	
	expr_corrs = pd.DataFrame(X_corr[genes].values,columns=index,index = index)
	clusters = expr_corrs.columns.get_level_values("Module")

	
	# pal = sns.husl_palette(len(set(clusters)), s=.9)
	pal = sns.color_palette("colorblind",len(set(clusters)))
	pal = sns.color_palette("hls", 8)
	

	clust_lut = dict(zip(list(set(clusters)), pal))
	clust_colors = pd.Series(clusters, index=expr_corrs.columns).map(clust_lut)
	
	g = sns.clustermap(expr_corrs, center=0, cmap="vlag",
		row_colors=clust_colors, col_colors=clust_colors,yticklabels=False,xticklabels=False,
		vmin=-1, vmax=1,
		dendrogram_ratio=(.1, .2))
	
	g.ax_row_dendrogram.remove()
	ax = g.ax_heatmap
	ax.set_xlabel("")
	ax.set_ylabel("")
	plt.tight_layout()
	
	g.savefig(f"../figs/{drug}/{tissue}/expression_heatmap.png",dpi=500,transparent = True)
	plt.close()
	sclr = StandardScaler()
	X2 = expression[['Run_ID']+genes]
	X2 = X2.merge(response, on = "Run_ID")
	X2 = X2.sort_values(by='Response')
	attrs = [X2['Response'].values,X2['Run_ID'].values]
	tuples = list(zip(*attrs))
	row_index = pd.MultiIndex.from_tuples(tuples, names=["Response", "Run_ID"])
	_X2 = sclr.fit_transform(np.log2(X2[genes].values+1))
	# _X2 = np.log2(X2[genes].values+1)
	df2 = pd.DataFrame(_X2,columns = index, index= row_index)
	
	resps = df2.index.get_level_values("Response")
	pal = sns.husl_palette(len(set(resps)),h=3,l=0.33, s=.8)
	pal = sns.color_palette("Set2",len(set(resps)))
	resp_lut = dict(zip(list(set(resps)), pal))
	# resp_lut = {1:'orange',0:'blue'}
	resp_colors = pd.Series(resps, index=df2.index).map(resp_lut)


	g = sns.clustermap(df2, center=0,cmap="vlag",
		row_colors=resp_colors, col_colors=clust_colors,yticklabels=False,xticklabels=False,
		row_cluster=False,col_linkage = row_linkage,
		dendrogram_ratio=(.1, .2))
	g.ax_row_dendrogram.remove()
	ax = g.ax_heatmap
	ax.set_xlabel("")
	ax.set_ylabel("")
	plt.tight_layout()
	g.savefig(f"../figs/{drug}/{tissue}/response_module_heatmap.png",dpi=300,transparent = True)
	plt.close()
















	X_corr.reset_index(names = "Gene",drop=False, inplace=True)
	print(X_corr)
	module_genes = {}
	for k in pd.unique(X_corr['cluster']):
		temp = X_corr[X_corr['cluster']==k]
		module_genes[k]=list(temp['Gene'].values)

	res_ = defaultdict(list)
	res_['Run_ID'] = expression['Run_ID'].values
	# sclr = StandardScaler()
	for module_num in sorted(module_genes.keys()):
		mg = module_genes[module_num]
		temp = expression[['Run_ID']+mg]
		av = np.mean(np.log2(temp[mg].values+1),axis=1)
		res_[f"Module {module_num}"] = av

	

	res_ = pd.DataFrame(res_)
	res_ = res_.merge(response, on = 'Run_ID')
	_r = res_.copy(deep=True)
	res_.drop(columns = ['Run_ID'],inplace=True)
	res_ = res_.melt(id_vars=['Response'],var_name = 'Module')
	
	ax = sns.boxplot(res_, y='value',x = 'Module',hue='Response')
	ax.set(ylabel = 'Module Score')
	plt.legend([],[], frameon=False)
	plt.tight_layout()
	plt.savefig(f"../figs/{drug}/{tissue}/module_scores.png",dpi=500)
	plt.close()
	ps = []
	for k in module_genes.keys():
		mod = f"Module {k}"
		temp = _r[[mod,'Response']]
		r_exp = temp[temp['Response']==1][mod].values
		nr_exp = temp[temp['Response']==0][mod].values
		u,p = mannwhitneyu(r_exp, nr_exp)
		ps.append(p)
	

	adj_ps = mt.multipletests(ps, method = 'fdr_bh')

	loo = LeaveOneOut()
	lr = Pipeline([('scale',StandardScaler()),
			('clf',LogisticRegression(max_iter = 10000, solver = 'liblinear', 
				class_weight = 'balanced'))])
	param_grid = {
		'clf__penalty':['l2'],
		'clf__C':[10**j for j in np.linspace(-5,1,50)]
	}

	pal = sns.color_palette("hls", 8)
	stat_str = []
	model = GridSearchCV(lr, param_grid)
	for k in sorted(module_genes.keys()):
		print("\n--------")
		print(k)
		stat_str.append(f"Module {k}")
		X = _r[f"Module {k}"].values.reshape(-1,1)
		y = _r['Response'].values
		bin_preds, prob_preds = [],[]
		for i, (train_index, test_index) in enumerate(loo.split(X)):
			X_train,y_train = X[train_index,:], y[train_index]
			X_test, y_test = X[test_index,:], y[test_index]
			model.fit(X_train, y_train)
			bin_preds.append(model.predict(X_test)[0])
			prob_preds.append(model.predict_proba(X_test)[0][1])
		stat_str.append(f"accuracy_score:\t {accuracy_score(y,bin_preds)}")
		stat_str.append(f"balanced_accuracy_score:\t {balanced_accuracy_score(y,bin_preds)}")
		stat_str.append(f"roc_auc_score\t {roc_auc_score(y,prob_preds)}")
		stat_str.append("--------\n")
		# print(accuracy_score(y,bin_preds))
		# print(balanced_accuracy_score(y,bin_preds))
		# print(roc_auc_score(y,prob_preds))
		fpr, tpr, thresholds = roc_curve(y, prob_preds, pos_label=1)		
		plt.plot(fpr,tpr,label = f"Module {k}",color = pal[k-1])
	plt.plot([0, 1], ls="--",color = 'black')
	plt.legend(loc="upper left")
	plt.xlabel("False Positive Rate")
	plt.ylabel("True Positive Rate")
	# plt.title("p value = {p}".format(p=pv))
	# plt.show()
	plt.savefig(f"../figs/{drug}/{tissue}/module_loo.png",dpi=500,transparent = True)
	with open(f"../figs/{drug}/{tissue}/module_loo_performance.txt","w") as ostream:
		ostream.writelines([x+"\n" for x in stat_str])
	plt.close()




		
		

	
	



if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	# parser.add_argument("-config",help="The config file for these experiments")
	# args = parser.parse_args()
	
	# with open(args.config) as file:
	# 	config = yaml.safe_load(file)

	main({})