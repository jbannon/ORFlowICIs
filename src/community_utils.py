import networkx as nx
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt 
import sys
import pandas as pd
import numpy as np
from cdlib import algorithms
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.cluster import v_measure_score, adjusted_rand_score, silhouette_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.preprocessing import StandardScaler
import community.community_louvain as community_louvain
from collections import defaultdict
from typing import List, Dict




def transform_and_detect_communities(
	G_:nx.Graph,
	source_field:str = 'curvature',
	transform:str = 'sigmoid',
	algorithm:str = 'leiden',
	community_field:str = 'community',
	sigmoid_constant:float=10.0):
	


	G = G_.copy()
	assert algorithm in ['leiden','louvain'], "algorithm must be one of 'leiden' or 'louvain' "

	if transform == 'sigmoid':
		for e in G.edges(data=True):
			e[2]['new_weight'] = sigmoid_constant/(1.+np.exp(-e[2][source_field]))
	else: 
		max_val = max(nx.get_edge_attributes(G, source_field).values())
		min_val = min(nx.get_edge_attributes(G, source_field).values())
		for e in G.edges(data=True):
			e[2]['new_weight'] = (e[2][source_field]-min_val)/(max_val-min_val)


	if algorithm =='leiden':
		coms = algorithms.leiden(G, weights='new_weight')
	elif algorithm == 'louvain':
		coms = algorithms.louvain(G, weight='weight', resolution=1., randomize=False)


	com_assignments = {}
	for j in range(len(coms.communities)):
	
		for node in coms.communities[j]:
			com_assignments[node]=j

	nx.set_node_attributes(G, com_assignments,community_field)


	return G, coms

def cut_graph_by_quantile(
	G_:nx.Graph,
	q:float,
	field:str = 'curvature'
	) ->nx.Graph:
	
	G = G_.copy()
	
	field_values = [edge[2][field] for edge in G.edges(data=True)]
	cut_value = np.quantile(field_values,q=q)

	drop_edges = [(e[0],e[1]) for e in G.edges(data=True) if e[2][field]<=cut_value]
	
	G.remove_edges_from(drop_edges)
	# print(nx.is_connected(G))
	
	return G, cut_value

def find_quantile_cutoffs(
	G_:nx.Graph,
	field:str='curvature',
	min_quantile:float = 0.15,
	max_quantile:float = 0.7,
	num_points:int = 100,
	drop_threshold:float = 0.001
	):

	G = G_.copy()
	modularities,cut_values = [],[]
	quantile_range = np.linspace(min_quantile,max_quantile,num_points)
	for quantile in quantile_range:
		
		_G, cut_value = cut_graph_by_quantile(G, quantile, field)
		
		clustering = {c: idx for idx, comp in enumerate(nx.connected_components(_G)) for c in comp}
		modularities.append(community_louvain.modularity(clustering, G, field))
		cut_values.append(cut_value)

	
	mod_last = modularities[0]
	qtiles, good_cuts = [],[]

	for i in range(1,len(modularities)):
		mod_curr = modularities[i]
		if mod_curr > mod_last > 1e-4 and abs(mod_last - mod_curr) / mod_last > drop_threshold:
			good_cuts.append(cut_values[i])
			qtiles.append(quantile_range[i])
		mod_last = mod_curr

	return good_cuts,qtiles



def cut_graph_by_weight(
	G_:nx.Graph,
	cutoff:float,
	weight_field:str = 'weight',
	)-> nx.Graph:
	
	G = G_.copy()
	drop_edges = [(edge[0],edge[1]) for edge in G.edges(data=True) if edge[2][weight_field]>cutoff]
	G.remove_edges_from(drop_edges)
	
	return G


def find_weight_cutoffs(
	G_:nx.Graph,
	weight_field:str = 'weight',
	step:float = 0.0025,
	drop_threshold:float = 0.001
	)->List[float]:
	

	G = G_.copy()

	max_weight = max(nx.get_edge_attributes(G, weight_field).values())
	modularities = []
	cutoff_range = np.arange(max_weight, 1, -step)

	for cutoff in cutoff_range:
		G = cut_graph_by_weight(G,cutoff,weight_field)
		clustering = {c: idx for idx, comp in enumerate(nx.connected_components(G)) for c in comp}
		modularities.append(community_louvain.modularity(clustering, G, weight_field))
	mod_last = modularities[-1]
	good_cuts = []

	for i in range(len(modularities) - 1, 0, -1):
		mod_curr = modularities[i]
		if mod_last > mod_curr > 1e-4 and abs(mod_last - mod_curr) / mod_last > drop_threshold:
			good_cuts.append(cutoff_range[i+1])
		mod_last = mod_curr

	return good_cuts


def assign_cc_labels(
	G:nx.Graph,
	min_comm_size:int = 5,
	X:np.array = None,
	id_to_idx:Dict = {},
	merge_coms:bool = False
	):

	num_ccs = len([x for x in nx.connected_components(G)])
	cc_sizes = [len(x) for x in nx.connected_components(G)]
	comp_map = {}
	for i, comp in enumerate(nx.connected_components(G)):
		for node in comp:
			comp_map[node] = i


	nx.set_node_attributes(G,comp_map,'community')
	return G, num_ccs, cc_sizes
