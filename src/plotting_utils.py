from collections import defaultdict
from matplotlib.pyplot import cm
import matplotlib as mpl 
import networkx as nx
import matplotlib.pyplot as plt 
from lifelines.statistics import logrank_test, multivariate_logrank_test, pairwise_logrank_test
from lifelines import KaplanMeierFitter
import seaborn as sns
import pandas as pd
import numpy as np 
from typing import List, Union
import sys 

def plot_multiple_survivals(
	G:nx.Graph,
	clinical_data:pd.DataFrame,
	drug:str,
	tissue:str,
	event_col:str = 'PFS_e',
	time_col:str = 'PFS_d',
	graph_id_field:str='Run_ID',
	graph_comm_field:str = 'community',
	clin_id_field:str = 'Run_ID',
	min_comm_size:int = 5,
	xlabel:str = 'Time (Days)',
	fname:str = None):
	
	results = defaultdict(list)
	for v in G.nodes(data=True):
		results[clin_id_field].append(v[1][graph_id_field])
		results['Community'].append(v[1][graph_comm_field])


	results = pd.DataFrame(results)
	results = results.merge(clinical_data,on=clin_id_field)
	

	community_sizes = results['Community'].value_counts()

	keep_communities = list(community_sizes.where(community_sizes>=min_comm_size).dropna().index)

	results = results[results['Community'].isin(keep_communities)]


	
	pw_test = pairwise_logrank_test(
		results[time_col],
		results['Community'],
		results[event_col]
		)
	
	ax = plt.subplot(111)
	kmf = KaplanMeierFitter()
		
	for com in sorted(pd.unique(results['Community'])):

		temp = results[results['Community']==com]
		kmf.fit(temp[time_col],temp[event_col],label = f"Group {com}")
		ax = kmf.plot_survival_function(ax=ax,ci_show=False)
	ax.set(xlabel = xlabel,ylabel= "Survival Probability")
	# plt.title(f"Kaplan Meier Curves {drug.title()} {tissue.upper()}")
	plt.tight_layout()
	# plt.savefig(f"{drug}_{tissue}.png")

	if fname is None:
		plt.show()
	else:
		plt.savefig(fname)
		plt.close()	

	
	
def analyze_multiple_survivals(
	G:nx.Graph,
	clinical_data:pd.DataFrame,
	event_col:str = 'PFS_e',
	time_col:str = 'PFS_d',
	graph_id_field:str='Run_ID',
	graph_comm_field:str = 'community',
	clin_id_field:str = 'Run_ID',
	min_comm_size:int = 5):

	results = defaultdict(list)
	for v in G.nodes(data=True):
		results[clin_id_field].append(v[1][graph_id_field])
		results['Community'].append(v[1][graph_comm_field])


	results = pd.DataFrame(results)
	results = results.merge(clinical_data,on = clin_id_field)

	community_sizes = results['Community'].value_counts()

	keep_communities = list(community_sizes.where(community_sizes>=min_comm_size).dropna().index)

	results = results[results['Community'].isin(keep_communities)]

	
	mv_test = multivariate_logrank_test(
		results[time_col],
		results['Community'],
		results[event_col]
		)
	pw_test = pairwise_logrank_test(
		results[time_col],
		results['Community'],
		results[event_col]
		)
	

	return mv_test, pw_test




def plot_colored_graph(
	G:nx.Graph,
	color_field:str = 'response',
	cm_name:str = 'Dark2'):

	groups = set(nx.get_node_attributes(G,color_field).values())
	cmap = mpl.colormaps[cm_name]
	colors = [cmap(k) for k in np.linspace(0,1,len(groups))]
	
	
	att_dict = nx.get_node_attributes(G, color_field)

	comm_map = defaultdict(list)

	for node in G.nodes():
		comm_map[att_dict[node]].append(node)
	

	pos = nx.spring_layout(G)
	for i in range(len(comm_map.keys())):
		comm = list(comm_map.keys())[i]
		nx.draw_networkx_nodes(G, pos=pos, nodelist=comm_map[comm],
               node_color=colors[i], label=comm,node_size=100)
	nx.draw_networkx_edges(G, pos=pos)
	plt.axis('off')
	plt.legend(scatterpoints = 1)
	plt.tight_layout()
	plt.show()
	plt.close()

def plot_and_save_colored_graph(
	G_:nx.Graph,
	color_field:str ='response',
	colors:Union[List[str],str] = ['purple','orange'],
	alpha:float = 0.5,
	fname:str = None,
	min_comm_size:float = 0):
	
	
	G = G_.copy()
	if isinstance(colors, str):
		groups = set(nx.get_node_attributes(G,color_field).values())
		cmap = mpl.colormaps[colors]
		colors = [cmap(k) for k in np.linspace(0,1,len(groups))]
	
	drop_nodes = []
	for cc in nx.connected_components(G):
		if len(cc)<min_comm_size:
			drop_nodes.extend([x for x in cc])
	G.remove_nodes_from(drop_nodes)

	att_dict = nx.get_node_attributes(G, color_field)
	

	comm_map = defaultdict(list)

	for node in G.nodes():
		comm_map[att_dict[node]].append(node)
	
	pos = nx.spring_layout(G)
	for i in range(len(comm_map.keys())):
		comm = list(comm_map.keys())[i]

		if color_field == 'tissue':
			nx.draw_networkx_nodes(G, pos=pos, nodelist=comm_map[comm],
           		node_color=colors[i], label=f"{comm.upper()}",node_size=200)
		else:
			nx.draw_networkx_nodes(G, pos=pos, nodelist=comm_map[comm],
           		node_color=colors[i], label=f"{comm}".title(),node_size=200)
	nx.draw_networkx_edges(G, pos=pos,alpha=alpha)
	plt.axis('off')
	plt.legend(scatterpoints = 1)
	plt.tight_layout()
	if fname is None:
		plt.show()
	else:
		plt.savefig(fname)
		plt.close()	

	

	

	