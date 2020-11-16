import numpy as np
import pandas as pd
import prince

def CorrespondanceAnalysis(df,theta=1,dimensions=3,coordinates='top',all_nodes=False):

	# Distinguishing coordinate edges from entities nodes
	entities = 'bottom' if coordinates=='top' else 'top'

	# Degree of entity nodes
	degrees = df.groupby(entities).count()[coordinates].sort_values(ascending=False)

	# filtering edge list to delete nodes that 
	df = df[df[entities].isin(degrees[degrees>=theta].index.values)]

	df['w']=1

	# Prototype version: Pivoting adjacency matrix in chunks
	# chunk_size = 100000
	# chunks = [x for x in range(0, df.shape[0], chunk_size)]
	# M = pd.concat([df.loc[df[entities].isin(degrees[degrees>=theta].index.values),[entities,coordinates,'w']].iloc[ chunks[i]:chunks[i + 1] - 1 ].pivot(index=entities, columns=coordinates, values='w') for i in range(0, len(chunks) - 1)])

	# Current version: pivoting adjacency matrix
	# Will fail for large sizes (long tailed degree distributions, theta=1)
	M = df[[entities,coordinates,'w']].pivot(index=entities,columns=coordinates,values='w')
	M.fillna(0,inplace=True)      

	# Selecting the core sub-graph that will be used in the CA
	M_mask = (~M.duplicated())
	selected_rows = M.loc[M_mask].index.values
	unselected_rows = M.loc[~M_mask].index.values

	# Correspondent Analysis
	ca = prince.CA(n_components=dimensions,
				   n_iter=4,copy=True,check_input=True,
				   engine='auto',random_state=np.random.randint(1,100))
	ca = ca.fit(M[M_mask])
	if all_nodes:
	    row_coords = ca.row_coordinates(M)
	else:
	    row_coords = ca.row_coordinates(M[M_mask])
	col_coords = ca.column_coordinates(M[M_mask])

	output=pd.concat([row_coords,col_coords],axis=0)

	info = {'explained_inertia':ca.explained_inertia_,}

	return output,info;
