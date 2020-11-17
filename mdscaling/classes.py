import numpy as np
import pandas as pd

# scaling methods
from .bipartite_scaling.factor_analysis import *


class DiBipartite():

	"""
	Directed bipartite graph, stored as a list of edges in a pandas DataFrame.
	Nodes are identified with a string label.
	Edge multiplicity is 1 (no repeated edges).
	Bipartite has two sets of nodes: top, bottom.
	All edges go from bottom to top.

	Storage convention in the CSV file is,
	for each row/edge:

	bottom_node_i,top_node_j

	"""

	edges_df = []
	top_nodes_list = []
	bottom_nodes_list = []
	nodes_list = []
	edges = np.nan
	top_nodes = np.nan
	bottom_nodes = np.nan

	info = []
	embedding = []

	def __init__(self,filename):

		# Loading the list of edges
		df = pd.read_csv(filename,header=None,names=['bottom','top'],dtype={'bottom':'str','top':'str'})

		# Checks
		if df.shape[0] == 0:
			raise ValueError('No edges in file.')
		if df.shape[1] > 2:
			raise ValueError('File has more than two columns. Format is csv of N_edges x 2.')
		if df.duplicated().sum()>0:
			print('Deleting were %d duplicated edges...'%(df.duplicated().sum()))
			df=df[~df.duplicated()]
		if df.isna().any().any():
			print('Deleting %d edges with NaN values...'%((df['bottom'].isna()|df['top'].isna()).sum()))
			df.dropna(axis=0,how='any',inplace=True)
		if np.intersect1d(df['bottom'],df['top']).size>0:
			raise ValueError('Top and bottom nodes are not disjoint sets.')
		if df.shape[0] == 0:
			raise ValueError('After deleting repeated and NaN edges, there were none left.')

		# storing values
		self.edges_df = df.copy(deep=True)
		del df
		self.top_nodes_list = self.edges_df['top'].unique().tolist()
		self.bottom_nodes_list = self.edges_df['bottom'].unique().tolist()
		self.nodes_list = self.top_nodes_list + self.bottom_nodes_list
		self.edges = self.edges_df.shape[0]
		self.top_nodes = len(self.top_nodes_list)
		self.bottom_nodes = len(self.bottom_nodes_list)

		return

	# Scaling methods
	def CA(self,theta=3,dimensions=3,coordinates='top',all_nodes=False):
		embedding,info = CorrespondanceAnalysis(self.edges_df,theta=theta,
												dimensions=dimensions,coordinates=coordinates,
												all_nodes=all_nodes)
		self.info = info
		self.embedding = embedding
		return;





