from sklearn.manifold import LocallyLinearEmbedding, MDS,TSNE
import os
import yaml
import argparse
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt 
import sys
import pandas as pd
import numpy as np
# from cdlib import algorithms
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, silhouette_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
import community.community_louvain as community_louvain
from collections import defaultdict
from typing import List,Dict, Union
import community_utils as cu
import utils
import ORCurvature as nc
import plotting_utils as pu
from sklearn.manifold import LocallyLinearEmbedding, MDS,TSNE
from sklearn.metrics.cluster import adjusted_rand_score


def main(config:Dict):

	## Directories:
	
	data_dir:str 
	geneset_dir:str 
	res_base:str 
	fig_base:str
	clinical_file:str

	data_dir,  geneset_dir, res_base, fig_base, clinical_file =\
		utils.unpack_parameters(config['DIRECTORIES'])
	

	## Drug and Tissue Params:

	drug:str
	tissue:str
	alpha:float 
	genesets:List[str]

	drug, tissue, alpha, genesets,de_thresh,trim =\
		utils.unpack_parameters(config['EXPERIMENT_PARAMS'])

	## Graph Parameters:
	

	num_neighbors:Union[int,str]
	do_log_transform:bool
	log_offset:float
	manifold:bool


	num_neighbors, do_log_transform, log_offset, manifold =\
		utils.unpack_parameters(config['GRAPH_PARAMS'])


	## Community Detection Params

	# Flow Params

	num_iters:int
	step_size:float

	# Comm Params

	min_size:int 
	drop_threshold:float
	weight_step:float 
	weight_field:str

	


	num_iters, step_size, min_size, drop_threshold, weight_step, weight_field =\
	 utils.unpack_parameters(config['COMMUNITY_PARAMS'])

	expression_file = f"../data/cri/{drug}/{tissue}/expression_full.csv"
	feature_file = f"../data/cri/{drug}/{tissue}/features.csv"
	imm_feat = ['Miracle','TIDE','TIDE_IFNG','TIDE_MSI','TIDE_CD274','TIDE_CD8','TIDE_CTL_Flag',
		'TIDE_Dysfunction',	'TIDE_Exclusion','TIDE_MDSC','TIDE_CAF','TIDE_TAM_M2',	'TIDE_CTL']
	
	do_features = False
	
	manif_string = "/manifold" if manifold else "/graph"
	trim_string = "/trimmed" if trim else ""
	feat_string = "/features" if do_features else ""
	
	
	res_dir = f"{res_base}/{drug}/{tissue}{feat_string}{trim_string}{manif_string}"
	fig_dir = f"{fig_base}/{drug}/{tissue}{feat_string}{trim_string}{manif_string}"

	os.makedirs(res_dir,exist_ok=True)
	os.makedirs(fig_dir,exist_ok=True)

	binarize:bool = True

	expression = pd.read_csv(expression_file)
	features = pd.read_csv(feature_file)
	features = features[['Run_ID']+imm_feat]
	
	

	clinical_data = pd.read_csv(clinical_file,sep = "\t")
	clinical_data = clinical_data[['Run_ID', 'TCGA_Tissue', 'Response',  'PFS_e', 'PFS_d']]#'OS_e', 'OS_d',
	clinical_data = clinical_data[clinical_data['TCGA_Tissue']!='GBM']
	clinical_data = clinical_data[clinical_data['Run_ID'].isin(expression['Run_ID'])]
	clinical_data.dropna(inplace=True)
	if drug == 'Pembro' and tissue == 'SKCM' and trim:
		drop_samples = ['Gide_Cell_2019-PD34-ar-951','Gide_Cell_2019-PD41-ar-958', 'Gide_Cell_2019-PD52-ar-970', 'Gide_Cell_2019-PD54-ar-972','Liu_NatMed_2019-p004-ar-00012']
		clinical_data = clinical_data[~clinical_data['Run_ID'].isin(drop_samples)]

	expression = expression[expression['Run_ID'].isin(clinical_data['Run_ID'].values)]

	features = features[features['Run_ID'].isin(clinical_data['Run_ID'].values)]
	genes = utils.fetch_union_genesets()


	

	expression = expression[['Run_ID']+[x for x in genes if x in expression.columns]]
	idx_to_id = {i:expression['Run_ID'].values[i] for i in range(len(expression['Run_ID'].values))}
	# id_to_idx = {i:expression['Run_ID'].values[i] for i in range(len(expression['Run_ID'].values))}
	id_to_resp, id_to_bin_resp, id_to_tissue = {}, {}, {}
	
	# print(clinical_data['TCGA_Tissue'].value_counts())
	# sys.exit(1)


	for idx, row in clinical_data.iterrows():
		resp = "responder" if row['Response'] in ['Partial Response','Complete Response'] else "non-responder"
		id_to_bin_resp[row['Run_ID']] = resp	
		id_to_resp[row['Run_ID']] = row['Response'] #resp
		id_to_tissue[row['Run_ID']] = row['TCGA_Tissue']
	


	if do_log_transform:
		X = np.log2(expression[expression.columns[1:]].values+log_offset)
	else:
		X = expression[expression.columns[1:]].values
	
	
	if do_features:
		X = features[features.columns[1:]].values
	if isinstance(num_neighbors,str):
		if num_neighbors.lower()=='auto':
			num_neighbors = int(np.ceil(X.shape[0]/10))
		else:
			#raise NotImplimentedError
			# default behavior
			num_neighbors = 5

	
	
	
	if manifold:

		best_comp:int
		best_error = np.inf
		for comp in np.arange(2,min(15,X.shape[1])):
			embedding = LocallyLinearEmbedding(n_neighbors = num_neighbors,n_components=comp)
			X_ = embedding.fit_transform(X)
			
			if embedding.reconstruction_error_<=best_error:
				best_comp = comp
				best_error = embedding.reconstruction_error_
		embedding = LocallyLinearEmbedding(n_neighbors = num_neighbors,n_components=best_comp)
		X = embedding.fit_transform(X)
		manif_lrn = [f"Best Number Components:\t{best_comp}\n",f"Best Error:\t {best_error}"]
		with open(f"{res_dir}/LLE_stats.txt","w") as ostream:
			ostream.writelines(manif_lrn)
		
	
	# scaler = StandardScaler()
	# X = scaler.fit_transform(X)
	G = utils.build_graph(X,num_neighbors,mode='connectivity')
	
	node_resp_labels = {n: id_to_resp[idx_to_id[n]] for n in G.nodes()}
	node_bin_resp_labels = {n: id_to_bin_resp[idx_to_id[n]] for n in G.nodes()}
	node_tissue_labels = {n: id_to_tissue[idx_to_id[n]] for n in G.nodes()}
	
	nx.set_node_attributes(G, node_resp_labels, "response")
	nx.set_node_attributes(G, node_bin_resp_labels, "binary_response")
	nx.set_node_attributes(G, node_tissue_labels, "tissue")

	# for n in G.nodes(data=True):
	# 	print(n)
	pu.plot_and_save_colored_graph(G,
		color_field = 'tissue', 
		colors = 'Dark2',
		fname = f"{fig_dir}/tissue_before.png")

	pu.plot_and_save_colored_graph(G,
		color_field = 'binary_response', 
		colors = ['purple','orange'],
		fname = f"{fig_dir}/binary_response_original.png")


	nx.set_node_attributes(G,idx_to_id,"Run_ID")
	# print(G)
	curvature_module = nc.ORCurvature(G,alpha)
	curvature_module.compute_ricci_flow(
		num_iters = num_iters,
		step_size = step_size,
		update_by_weight=True)

	G_ = curvature_module.G.copy()

	cuts = cu.find_weight_cutoffs(G_)
	if len(cuts)==0:
		G_ = cu.cut_graph_by_weight(G_,1.05)
	else:
		G_ = cu.cut_graph_by_weight(G_,cuts[-1])
	
	G_, num_ccs, cc_sizes = cu.assign_cc_labels(G_,
		min_comm_size = min_size,
		merge_coms = False,
		X = X,
		id_to_idx = idx_to_id)

	
	labels = nx.get_node_attributes(G_, "Run_ID")

	true_response = nx.get_node_attributes(G_,"binary_response")
	
	true_response = list(map(lambda x: 1 if x =='responder' else 0,true_response.values()))
	comm_labels = nx.get_node_attributes(G_,"community")
	comm_labels = list(comm_labels.values())
	

	ari = adjusted_rand_score(true_response,comm_labels)
	with open(f"{res_dir}/adjusted_rand_score.txt", "w") as ostream:
		ostream.writelines([str(ari)])
	
	
	# for cc in nx.connected_components(G_):
	# 	print(len(cc))
	# 	cc_labs = [labels[n] for n in cc]
	# 	print(cc_labs)
	# 	print(clinical_data[clinical_data['Run_ID'].isin(cc_labs)])
	

	
	# pu.plot_colored_graph(G_)
	pu.plot_and_save_colored_graph(G_,
		color_field = 'tissue', 
		colors = 'Dark2',
		fname = f"{fig_dir}/tissue.png")

	pu.plot_and_save_colored_graph(G_,
		color_field = 'binary_response', 
		colors = ['purple','orange'],
		fname = f"{fig_dir}/binary_response_cut_graph.png")

	pu.plot_and_save_colored_graph(G_,
		color_field = 'binary_response', 
		colors = ['purple','orange'],
		fname = f"{fig_dir}/binary_response_cut_graph_trimmed.png",
		min_comm_size=min_size)

	

	
	pu.plot_and_save_colored_graph(
		G_,
		color_field = 'response', 
		colors = 'Dark2',
		fname = f"{fig_dir}/response_cut_graph.png")
	
	pu.plot_multiple_survivals(
		G_,
		clinical_data,
		drug,tissue,
		min_comm_size=min_size,
		fname = f"{fig_dir}/survival_curves.png")

	mv, pw = pu.analyze_multiple_survivals(
		G_,
		clinical_data)

	pw_res = f"Pairwise Logrank p-value:= {pw.p_value}\n"
	with open(f"{res_dir}/pvalue.txt","w") as ostream:
		ostream.writelines(pw_res)
	

	
	community_df = defaultdict(list)

	for node in G_.nodes(data=True):
		community_df['Run_ID'].append(node[1]['Run_ID'])
		community_df['Community'].append(node[1]['community'])

	community_df = pd.DataFrame(community_df)
	comm_counts = community_df['Community'].value_counts()
	keep_coms = list(comm_counts[comm_counts>min_size].index)
	
	community_df.to_csv(f"{res_dir}/community_assignments.csv",index=False)

	de_comms = community_df[community_df['Community'].isin(keep_coms)]
	de_comms.to_csv("./col_data.csv",index=False)

	sample_order = de_comms['Run_ID']
	if not do_features:
		count_matrix = pd.read_csv(f"../data/cri/{drug}/{tissue}/counts_full.csv")
		count_matrix.set_index('Run_ID',inplace = True)
		count_matrix = count_matrix.loc[sample_order,:]
		count_matrix.reset_index(inplace = True, drop = False, names = ['Run_ID'])
		count_matrix = count_matrix.transpose()
		count_matrix.reset_index(inplace = True)
		count_matrix.columns = count_matrix.iloc[0]
		count_matrix.drop(index=0,axis=0,inplace=True)
		count_matrix.rename(columns = {'Run_ID':'Gene_ID'},inplace = True)
		count_matrix.to_csv("./count_data.csv",index = False)

		cmd = f"Rscript compute_DE_genes.R {drug} {tissue} ./count_data.csv ./col_data.csv {de_thresh} {manif_string[1:]} {str(trim).upper()}"
		os.system(cmd)
		os.remove("./count_data.csv")
		os.remove("./col_data.csv")






if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-config",help="The config file for these experiments")
	args = parser.parse_args()
	
	with open(args.config) as file:
		config = yaml.safe_load(file)

	main(config)

