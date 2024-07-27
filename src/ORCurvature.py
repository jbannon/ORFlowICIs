from functools import lru_cache
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import sys
import numpy as np
import networkx as nx
import networkit as nit
import ot
import tqdm 
import time
import multiprocessing as mp


_G = nit.graph.Graph()
_alpha = 0.5
_weight = "weight"
_epsilon:float = 1.0
_num_processes = mp.cpu_count()
_min_degree:float = 1E-5
_min_distance:float = 1E-7
_APSP_Matrix:np.ndarray = {}
_cache_size = 1000000
_node_to_idx = {}





def _make_all_pairs_shortest_path_matrix(
	) -> np.ndarray:
	

	global _G

	
	D = np.array(nit.distance.APSP(_G).run().getDistances())
	
	return D


@lru_cache(_cache_size)
def _assign_single_node_density(node:int):
	
	
	neighbors_and_weights = []
	neighbors = list(_G.iterNeighbors(node))
	for neighbor in neighbors:
		edge_weight = _G.weight(node, neighbor)
		neighbors_and_weights.append((neighbor,edge_weight))
	
	node_degree = sum([x[1] for x in neighbors_and_weights])
	# print(neighbors_and_weights)

	nbrs = [x[0] for x in neighbors_and_weights]
	
	if node_degree>_min_degree:
		pmf = [(1.0-_alpha)*w/node_degree for _,w in neighbors_and_weights]
		labeled_pmf = [(nbr,(1.0-_alpha)*w/node_degree) for nbr, w in neighbors_and_weights]
	else:
		# assign equal weight to all neighbors
		pmf = [(1.0-_alpha)/len(neighbors_and_weights)]*len(neighbors_and_weights)
		labeled_pmf = [(nbr,(1.0-_alpha)/len(neighbors_and_weights)) for nbr, w in neighbors_and_weights]
		
	return pmf + [_alpha], nbrs + [node]




def _compute_single_edge_curvature(
	node1:int, 
	node2:int)-> Dict[Tuple[int,int],float]:

	pmf_x, nbrs_x = _assign_single_node_density(node1)

	pmf_y, nbrs_y = _assign_single_node_density(node2)
	# print(node1)
	# print(node2)
	
	
	D = _APSP_Matrix[np.ix_(nbrs_x, nbrs_y)]
	
		
	# node1_to_node2_distance, _ = nx.bidirectional_dijkstra(G, node1,node2)
	# node1_to_node2_distance = nit.distance.BidirectionalDijkstra(_G, node1, node2).run().getDistance()
	node1_to_node2_distance = _APSP_Matrix[node1,node2]
	OT_distance = ot.emd2(pmf_x, pmf_y,D)
	# print("between node {u} and node {v} the W1 distance is: {w}".format(u = node1,v = node2, w= OT_distance))
	
	if node1_to_node2_distance < _min_distance:
		kappa = 0
	else:
		kappa = 1 - OT_distance/node1_to_node2_distance


	return {(node1,node2):kappa}

def _single_edge_curvature_wrapper(curve_args):
	return _compute_single_edge_curvature(*curve_args)


def _compute_all_edge_curvatures(
	G:nx.Graph,
	alpha:float =0.5,
	weight_field:str = 'weight',
	num_processes:int = mp.cpu_count()
	):

	global _G
	global _alpha 
	global _weight
	global _APSP_Matrix
	global _num_processes
	global _node_to_idx
	
	if not nx.get_edge_attributes(G, weight_field):
		for (v1, v2) in G.edges():
			G[v1][v2][weight_field] = 1.0
	
	_G = nit.nxadapter.nx2nk(G,weightAttr=weight_field)
	
	_weight = weight_field
	_alpha = alpha
	_num_processes = num_processes
	_APSP_Matrix = _make_all_pairs_shortest_path_matrix()
	
	
	nx_to_nit, nit_to_nx = {},{}

	
	for node_idx, node in enumerate(G.nodes):
		nx_to_nit[node] = node_idx
		nit_to_nx[node_idx] = node

	_node_to_idx = nx_to_nit


	dispatchable_edges = [
		(nx_to_nit[u], nx_to_nit[v]) for u, v in G.edges()
		]


	with mp.get_context('fork').Pool(processes = _num_processes) as dispatcher:
		chunksize, extra= divmod(len(dispatchable_edges), _num_processes*4)
		if extra:
			chunksize+=1
		

		results = dispatcher.imap_unordered(_single_edge_curvature_wrapper, dispatchable_edges, chunksize=chunksize)
		dispatcher.close()
		dispatcher.join()

	curvatures = {}
	for edge_curvature in results:
		for edge in list(edge_curvature.keys()):
			curvatures[(nit_to_nx[edge[0]], nit_to_nx[edge[1]])] = edge_curvature[edge]


	return curvatures

def _compute_and_assign_curvatures(
	G:nx.Graph,
	alpha:float =0.5,
	weight_field:str = 'weight',
	num_processes:int = mp.cpu_count()
	)->nx.Graph:

	curvatures = _compute_all_edge_curvatures(
			G, 
			alpha,
			weight_field,
			num_processes)
	nx.set_edge_attributes(G, curvatures, 'curvature')

	return G


	
def _compute_ricci_flow(
	G:nx.Graph,
	num_iters:int = 10,
	step_size:float = 1.0,
	alpha:float = 0.5,
	update_by_weight:bool = False,
	weight_field:str = 'weight',
	num_processes:int = mp.cpu_count(),
	delta:float = 1E-4
	)->nx.Graph:
	
	# if not nx.is_connected(G):
	# 	G = nx.Graph(G.subgraph(max(nx.connected_components(G), key=len)))

	global _APSP_Matrix

	normalized_weight = float(len(G.edges()))


	if not nx.get_edge_attributes(G, "original_curvature"):
		G = _compute_and_assign_curvatures(G, 
			alpha = alpha, 
			weight_field=weight_field,
			num_processes = num_processes)

		for (u, v) in G.edges():
			G[u][v]["original_curvature"] = G[u][v]["curvature"]

	for T in range(num_iters):
		for (u, v) in G.edges():
			u_idx, v_idx = _node_to_idx[u],_node_to_idx[v]
			if update_by_weight:
				G[u][v][weight_field] -= step_size * (G[u][v]["curvature"]) * G[u][v][weight_field]
			else:
				G[u][v][weight_field] -= step_size * (G[u][v]["curvature"]) * _APSP_Matrix[u_idx,v_idx]
		
		
		edge_weights = nx.get_edge_attributes(G, weight_field)
		sum_edge_weights = sum(edge_weights.values())
		for k, v in edge_weights.items():
			edge_weights[k] = edge_weights[k] * (normalized_weight / sum_edge_weights)
		

		rc = nx.get_edge_attributes(G, "curvature")
		diff = max(rc.values()) - min(rc.values())
		
		if diff < delta:
			# logger.trace("Ricci curvature converged, process terminated.")
			break

		nx.set_edge_attributes(G, values=edge_weights, name=weight_field)

		G = _compute_and_assign_curvatures(G, 
				alpha = alpha, 
				weight_field=weight_field,
				num_processes = num_processes)


	return G



class ORCurvature:
	def __init__(
		self,
		G:nx.Graph,
		alpha:float=0.5,
		weight_field:str = 'weight',
		num_proc:int = mp.cpu_count()
		):

		self.G = G.copy()
		self.alpha = alpha 
		self.weight_field = weight_field
		self.num_proc = num_proc
		if not nx.get_edge_attributes(self.G, weight_field):
			for (v1, v2) in self.G.edges():
				self.G[v1][v2][weight_field] = 1.0
		# print(self.G)
	
	def compute_ricci_curvatures(
		self
		)->None:

		curvatures = _compute_all_edge_curvatures(
			self.G, 
			self.alpha,
			self.weight_field,
			self.num_proc)

		nx.set_edge_attributes(self.G, curvatures, 'curvature')


	def compute_ricci_flow(
		self,
		num_iters:int = 10,
		step_size:float = 1.0,
		update_by_weight:bool = False
		)->None:

		self.G = _compute_ricci_flow(
			G=self.G, 
			num_iters=num_iters,
			step_size=step_size,
			update_by_weight = update_by_weight,
			alpha = self.alpha,
			weight_field = self.weight_field,
			num_processes = self.num_proc)



if __name__ == '__main__':
	G = nx.Graph()
	# A nbhd
	G.add_node(0,letter = "w")
	G.add_node(1,letter = "a")
	G.add_node(2,letter = "x")
	G.add_edge(0,1, weight = 4)
	G.add_edge(1,2, weight = 2)

	# B neighborhood
	G.add_node(3, letter = "b")
	G.add_node(4, letter = "y")
	G.add_node(5,letter = "z")
	G.add_edge(3,4,weight = 5)
	G.add_edge(3,5,weight = 1)
	#Connect them
	G.add_edge(1,3,weight = 3)


	


	# nx.draw(G)
	# plt.show()
	# _compute_all_edge_curvatures(G,'weight',0.5)
	# rc.compute_curvatures()
	rc = ORCurvature(G,alpha=0.5)
	rc.compute_ricci_curvatures()

	G_ = rc.G.copy()
	for e in G_.edges(data=True):
		print(e)
	