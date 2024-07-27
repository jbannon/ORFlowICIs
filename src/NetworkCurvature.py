from typing import Dict, Tuple
import matplotlib.pyplot as plt
import sys
import numpy as np
import networkx as nx
import ot
import tqdm 
import time


# @lru_cache(1000000)
def _make_all_pairs_shortest_path_matrix(
	G:nx.Graph,
	weight:str = 'weight'
	) -> np.ndarray:
	
	s = time.time()
	N = len(G.nodes)
	D = np.zeros((N,N))

	path_lengths = dict(nx.all_pairs_dijkstra_path_length(G,weight=weight))
	for node1 in path_lengths.keys():
		node1Lengths = path_lengths[node1]
		for node2 in node1Lengths.keys():
			D[node1,node2] = np.round(node1Lengths[node2],5)
	e = time.time()
	# print("apsp took {t}".format(t=e-s))
	if not (D==D.T).all():
		issues = np.where(D!=D.T)
		print("symmetry issue")
		sys.exit(1)

	return D



def _assign_single_node_density(
	G:nx.Graph,
	node:int,
	weight:str = 'weight',
	alpha:float = 0.5,
	measure_name:str = 'density',
	min_degree_value:float = 1E-5
	) -> None:
	
	


	neighbors_and_weights = []
	for neighbor in G.neighbors(node):
		edge_weight = G[node][neighbor][weight]
		neighbors_and_weights.append((neighbor,edge_weight))
	
	node_degree = sum([x[1] for x in neighbors_and_weights])
	nbrs = [x[0] for x in neighbors_and_weights]

	if node_degree>min_degree_value:
		
		pmf = [(1.0-alpha)*w/node_degree for _,w in neighbors_and_weights]
		labeled_pmf = [(nbr,(1.0-alpha)*w/node_degree) for nbr, w in neighbors_and_weights]
	else:
		# assign equal weight to all neighbors
		pmf = [(1.0-alpha)/len(neighbors_and_weights)]*len(neighbors_and_weights)
		labeled_pmf = [(nbr,(1.0-alpha)/len(neighbors_and_weights)) for nbr, w in neighbors_and_weights]
	
	nx.set_node_attributes(G,{node:{x:y for x,y in labeled_pmf}},measure_name)
	
	
	return pmf + [alpha], nbrs + [node]
	


def _compute_single_edge_curvature(:
	G:nx.Graph,
	node1:int, 
	node2:int, 
	weight:str = "weight",
	alpha:float = 0.5,
	measure_name:str = 'density',
	min_degree_value:float = 1E-5,
	min_distance:float = 1E-7,
	path_method:str = "all_pairs",
	APSP_Matrix = None,
	sinkhorn_thresh:int = 2000,
	epsilon:float = 1
	)-> Tuple[Tuple[int,int],float]:
	

	# print("in single edge")

	pmf_x, nbrs_x = _assign_single_node_density(G,node1,weight, alpha, measure_name, min_degree_value)
	pmf_y, nbrs_y = _assign_single_node_density(G,node2, weight, alpha, measure_name, min_degree_value)
	
	if path_method == "all_pairs" and APSP_Matrix is None:
		path_method = "pairwise"
	
	if path_method == 'pairwise':
		D = []
		for x_nbr in nbrs_x:
			temp_distances = []
			for y_nbr in nbrs_y:
				distance, path = nx.bidirectional_dijkstra(G, x_nbr,y_nbr)
				temp_distances.append(distance)
			D.append(temp_distances)
		D = np.array(D)
	else:
		D = APSP_Matrix[np.ix_(nbrs_x, nbrs_y)]

	
	node1_to_node2_distance, _ = nx.bidirectional_dijkstra(G, node1,node2)


	if max(D.shape[0],D.shape[1])>=sinkhorn_thresh:
		OT_distance = ot.sinkhorn2(pmf_x, pmf_y, D,epsilon)
	else:
		OT_distance = ot.emd2(pmf_x, pmf_y,D)

	# print("between node {u} and node {v} the W1 distance is: {w}".format(u = node1,v = node2, w= OT_distance))
	if node1_to_node2_distance < min_distance:
		kappa = 0
	else:
		kappa = 1 - OT_distance/node1_to_node2_distance
	

	return ((node1,node2),kappa)

	


def _compute_node_curvatures(
	G:nx.Graph,
	weight:str = 'weight',
	measure_name:str = 'density',
	edge_curve:str = 'ricci_curvature',
	node_curve:str = 'node_curvature',
	norm_node_curve:str  = 'node_curvature_normalized'
	)->None:
	
	for node, measure in G.nodes(data = True):
		
		normalized_curvature = 0 
		raw_curvature = 0 
		local_measure = measure[measure_name]
		

		for x in G.neighbors(node):
			term_weight = local_measure[x]

			sum_term = G[node][x][edge_curve]
			raw_curvature += sum_term
			normalized_curvature += sum_term*term_weight
		nx.set_node_attributes(G,{node:raw_curvature}, node_curve)
		nx.set_node_attributes(G,{node:normalized_curvature},norm_node_curve)


class OllivierRicciCurvature:
	def __init__(
		self,
		G:nx.Graph,
		alpha:float = 0.5,
		weight_field:str = "weight",
		path_method:str = "pairwise",
		curvature_field:str = "ricci_curvature",
		node_field:str = "node_curvature",
		norm_node_field:str = "node_curvature_normalized",
		measure_name:str = 'density',
		min_distance:float = 1E-5,
		min_degree:float = 1E-5,
		sinkhorn_thresh:int = 2000,
		epsilon:float = 1.0
		) ->None:
		
		self.G = G.copy()
		self.alpha = alpha
		self.weight_field = weight_field
		self.path_method = path_method
		self.curvature_field = curvature_field
		self.node_field = node_field
		self.norm_node_field = norm_node_field
		self.measure_name = measure_name
		self.min_distance = min_distance
		self.min_degree = min_degree
		self.sinkhorn_thresh = sinkhorn_thresh
		self.epsilon = epsilon
		self.edge_curvatures_computed = False
		self.node_curvatures_computed = False
		self.APSP_Matrix = None

		if not nx.get_edge_attributes(self.G, self.weight_field):
			for (v1, v2) in self.G.edges():
				self.G[v1][v2][self.weight_field] = 1.0


		if self.path_method == 'all_pairs':
			self.APSP_Matrix = _make_all_pairs_shortest_path_matrix(self.G,self.weight_field).copy()

		self.verbose = True

	def compute_edge_curvatures(
		self
		) -> None:
		
		curvatures = {}
		for edge in (tqdm.tqdm(self.G.edges(),leave=False) if self.verbose else self.G.edges()):
			node1, node2 = edge[0],edge[1]
			
			curv_tuple = _compute_single_edge_curvature(
				G = self.G,
				node1 = node1,
				node2 = node2,
				weight = self.weight_field,
				alpha = self.alpha,
				measure_name = self.measure_name,
				min_degree_value = self.min_degree,
				min_distance = self.min_distance,
				path_method = self.path_method,
				APSP_Matrix = self.APSP_Matrix,
				sinkhorn_thresh = self.sinkhorn_thresh,
				epsilon = self.epsilon
				)

			curvatures[curv_tuple[0]] = curv_tuple[1]
		nx.set_edge_attributes(self.G, curvatures, self.curvature_field)
		self.edge_curvatures_computed = True

	
	def compute_node_curvatures(self) -> None:
		if not self.edge_curvatures_computed:
			self.compute_edge_curvatures()
		
		_compute_node_curvatures(
			G = self.G,
			weight = self.weight_field,
			measure_name= self.measure_name,
			edge_curve = self.curvature_field,
			node_curve = self.node_field,
			norm_node_curve = self.norm_node_field
			)
		self.node_curvatures_computed = True
	
	def compute_curvatures(self)-> None:
		self.compute_edge_curvatures()
		self.compute_node_curvatures()



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

	


	# nx.draw(G,labels = {node:})
	# plt.show()
	rc  = OllivierRicciCurvature(G)
	rc.compute_curvatures()
	G_ = rc.G.copy()
	for e in G_.edges(data = True):
		print(e)

	G.add_edge(1,5,weight = 1)
	G.add_edge(0,4, weight = 0.5)
	rc  = OllivierRicciCurvature(G)
	rc.compute_curvatures()
	G_ = rc.G.copy()
	for e in G_.edges(data = True):
		print(e)




