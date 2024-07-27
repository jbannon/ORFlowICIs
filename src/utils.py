import pandas as pd
from typing import List,Dict
import numpy as np 
import networkx as nx
from sklearn.neighbors import kneighbors_graph


def unpack_parameters(
	D:Dict
	):
	
	if len(D.values())>1:
		return tuple(D.values())
	else:
		return tuple(D.values())[0]


def build_graph(
	X:np.array,
	n_nbrs:int,
	mode: str = 'connectivity',
	include_self:bool = False
	)->nx.Graph:
	

	A = kneighbors_graph(X, n_nbrs, mode=mode, include_self=include_self).toarray()
	for i in range(A.shape[0]):
		for j in range(i+1, A.shape[1]):
			if A[i,j] >0 or A[j,i] >0:
				edge_weight = max(A[i,j],A[j,i])
				A[i,j] = edge_weight
				A[j,i] = edge_weight

	G = nx.from_numpy_array(A)

	return G


def fetch_geneset(
	geneset_name:str
	)->List[str]:
	
	gsn_2_file = {
		'cosmic1':'cosmic_1.txt',
		'cosmic2':'cosmic_2.txt',
		'cosmic':'cosmic_all.txt',
		'kegg':'kegg.txt',
		'vogelstein':'vogelstein.txt',
		'auslander':'auslander.txt',
		'mdsig':'mdsig_hallmarks.txt'
		}
	geneset_name = geneset_name.lower()
	
	gs_file = f"../data/genesets/{gsn_2_file[geneset_name]}"

	with open(gs_file,"r") as istream:
		lines = istream.readlines()

	gene_names = [x.rstrip() for x in lines]
	return gene_names


def fetch_union_genesets(
	geneset_base:str = "../data/genesets",
	genesets:List[str] = ['kegg','auslander','vogelstein'],
	)->List[str]:
	
	all_genes = []
	gsn_2_file = {
		'cosmic1':'cosmic_1.txt',
		'cosmic2':'cosmic_2.txt',
		'cosmic':'cosmic_all.txt',
		'kegg':'kegg.txt',
		'vogelstein':'vogelstein.txt',
		'auslander':'auslander.txt',
		'mdsig':'mdsig_hallmarks.txt'
		}
	
	for gs in genesets:
		gs_file = f"{geneset_base}/{gsn_2_file[gs]}"
		with open(gs_file,"r") as istream:
			lines = istream.readlines()
		gene_names = [x.rstrip() for x in lines]

		all_genes.extend(gene_names)


	all_genes = sorted(list(set(all_genes)))
	return all_genes

def fetch_embeddings(
	embedding_file:str
	)->Dict[str,np.ndarray]:
	

	with open(embedding_file,"r") as istream:
		vecs = istream.readlines()
	gene_2_vec = {}
	for vec in vecs:
		split_vec = vec.rstrip().split(" ")
		gene = split_vec[0]
		vec = np.array([float(x) for x in split_vec[1:]])
		gene_2_vec[gene] = vec

	return gene_2_vec

def embed_patients(
	sample_data:pd.DataFrame,
	mutation_data:pd.DataFrame,
	vaf_cutoff:float,
	gene_embeddings:Dict[str,np.ndarray],
	pooling:str = 'avg'
	)->np.ndarray:
	
	patient_embeddings = []

	for idx, row in sample_data.iterrows():
		patient_sample_id = row['SAMPLE_ID']

		this_patient_mutations = mutation_data[(mutation_data['Tumor_Sample_Barcode']==patient_sample_id) & \
			(mutation_data['tumor_vaf']>=vaf_cutoff)]
	
		mutated_genes = this_patient_mutations['Hugo_Symbol'].values

		num_mutations = len(mutated_genes)
	
		patient_vectors = []
		
		for gene in mutated_genes:
			if gene in gene_embeddings.keys():
				patient_vectors.append(gene_embeddings[gene])
	
		patient_vectors = np.array(patient_vectors)

		if pooling =='avg':
			pooled_embedding = np.mean(patient_vectors,axis=0)
		elif pooling == 'max':
			pooled_embedding = np.max(patient_vectors,axis=0)
		elif pooling == 'min':
			pooled_embedding = np.min(patient_vectors,axis=0)
		elif pooling == 'med':
			pooled_embedding = np.median(patient_vectors,axis=0)
		
		patient_embeddings.append(pooled_embedding)

	return np.array(patient_embeddings)



