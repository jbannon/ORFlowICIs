from cdlib import algorithms
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from itertools import count
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List
import os
import pydot
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt 
import sys
import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, silhouette_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
# from cdlib import algorithms
from cdlib import algorithms, viz
import utils
from collections import defaultdict
import lifelines
from lifelines import KaplanMeierFitter
import seaborn as sns
from lifelines.statistics import logrank_test
import tqdm
import ORCurvature as nc

def main(config:Dict={}):

	drug:str = 'Nivo'
	tissue:str = 'PANCAN'

	fig_dir = f"../figs/{drug}/{tissue}/"
	res_dir = f"../results/{drug}/{tissue}/"
	os.makedirs(fig_dir,exist_ok=True)
	os.makedirs(res_dir,exist_ok=True)

	expression_file = f"../data/cri/{drug}/{tissue}/expression_full.csv"
	expression = pd.read_csv(expression_file)

	clinical_data = pd.read_csv("../data/cri/iatlas-ici-sample_info.tsv",sep = "\t")
	clinical_data = clinical_data[['Run_ID', 'TCGA_Tissue', 'Response',  'PFS_e', 'PFS_d']]#'OS_e', 'OS_d',
	clinical_data = clinical_data[clinical_data['Run_ID'].isin(expression['Run_ID'])]
	clinical_data.dropna(inplace=True)
	expression = expression[expression['Run_ID'].isin(clinical_data['Run_ID'].values)]
	
	# clinical_data.to_csv(f"{drug}_{tissue}.csv")
	genes = utils.fetch_union_genesets()
	

	expression = expression[['Run_ID']+[x for x in genes if x in expression.columns]]
	idx_to_id = {i:expression['Run_ID'].values[i] for i in range(len(expression['Run_ID'].values))}
	# id_to_idx = {i:expression['Run_ID'].values[i] for i in range(len(expression['Run_ID'].values))}
	id_to_resp = {}
	
	for idx, row in clinical_data.iterrows():
		resp = "responder" if row['Response'] in ['Partial Response','Complete Response'] else "non-responder"
		id_to_resp[row['Run_ID']] = row['Response'] #resp
	

	X = np.log2(expression[expression.columns[1:]].values+1)
	max_k = int(np.ceil(X.shape[0]/5))
	

	q_values = np.round(10**np.linspace(-6,0,100),3)

	results = defaultdict(list)
	connectivity = defaultdict(list)
	
	for k in tqdm.tqdm(range(2,max_k+1)):

		G = utils.build_graph(X,k)
		node_resp_labels = {n: id_to_resp[idx_to_id[n]] for n in G.nodes()}

		nx.set_node_attributes(G, node_resp_labels, "response")
		nx.set_node_attributes(G,idx_to_id,"Run_ID")
		
		connectivity['Num. Neighbors'].append(k)
		connectivity['Connected'].append(nx.is_connected(G))

		for alpha in tqdm.tqdm([0, 0.25, 0.5, 0.75, 0.99,1],leave=False):
			G_ = G.copy()
			
			orc = nc.ORCurvature(G_,alpha)
			orc.compute_ricci_curvatures()
			
			for q in tqdm.tqdm(q_values,leave= False):
				
				new_G = orc.G.copy()
				new_G = utils.cut_by_curvature_quantile(new_G,q,'curvature')
				ccs = list(nx.connected_components(new_G))
				num_ccs = len(ccs)
				lcc_size = np.max([len(cc) for cc in ccs])
				scc_size= np.min([len(cc) for cc in ccs])
				results['Num. Neighbors'].append(k)
				results['Alpha'].append(alpha)
				results['Quantile'].append(q)
				results['Num. Components'].append(num_ccs)
				results['Largest CC Size'].append(lcc_size)
				results['Smallest CC Size'].append(scc_size)
				
	

	connectivity = pd.DataFrame(connectivity)
	results = pd.DataFrame(results)
	connectivity.to_csv(f"{res_dir}connectivity.csv",index=False)
	results.to_csv(f"{res_dir}cut_stats.csv",index=False)

	results = defaultdict(list)
	for k in tqdm.tqdm(range(2,max_k+1)):

		G = utils.build_graph(X,k)
		node_resp_labels = {n: id_to_resp[idx_to_id[n]] for n in G.nodes()}

		nx.set_node_attributes(G, node_resp_labels, "response")
		nx.set_node_attributes(G,idx_to_id,"Run_ID")
		
		# connectivity['Num. Neighbors'].append(k)
		# connectivity['Connected'].append(nx.is_connected(G))

		for alpha in tqdm.tqdm([0, 0.25, 0.5, 0.75, 0.99,1],leave=False):
			for num_iters in tqdm.tqdm(np.arange(10,51,5),leave=False):
				G_ = G.copy()
			
				orc = nc.ORCurvature(G_,alpha)
				orc.compute_ricci_flow(num_iters = num_iters)
			
				for q in tqdm.tqdm(q_values,leave= False):
				
					new_G = orc.G.copy()
					new_G = utils.cut_by_curvature_quantile(new_G,q,'curvature')
					ccs = list(nx.connected_components(new_G))
					num_ccs = len(ccs)
					lcc_size = np.max([len(cc) for cc in ccs])
					scc_size= np.min([len(cc) for cc in ccs])
					results['Num. Neighbors'].append(k)
					results['Num. Iters'].append(num_iters)
					results['Alpha'].append(alpha)
					results['Quantile'].append(q)
					results['Num. Components'].append(num_ccs)
					results['Largest CC Size'].append(lcc_size)
					results['Smallest CC Size'].append(scc_size)

	results = pd.DataFrame(results)
	results.to_csv(f"{res_dir}flow_cut_stats.csv",index=False)


if __name__ == '__main__':
	main()


