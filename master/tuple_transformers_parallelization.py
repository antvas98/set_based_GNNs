import torch
from torch_geometric.data import (InMemoryDataset, Data)
import time
import numpy as np
import networkx as nx
import itertools
from tqdm import tqdm
from joblib import Parallel, delayed


def tuple_structure(tup,graph):
    k = len(tup)
    structure = np.zeros((k,k))

    for i, j in itertools.combinations(range(len(tup)), 2): # if i want to allow self loops  --> with_replacement
      if tup[i] == tup[j]:
        structure[i,j] = 2
        structure[j,i] = 2
      elif (tup[i],tup[j]) in graph.edges: 
          structure[i,j] = 1
          structure[j,i] = 1
      
    return np.triu(structure,k=1).flatten()

def tuple_k_local_transformer_iso(data,k=2):
  data.edge_attr = None
  graph = nx.Graph(directed=False)

  for i in range(data.x.shape[0]):
      node_attr = data.x[i,:].cpu().detach().numpy()
      graph.add_node(i,vector=node_attr)

  # Add edges (without their attributes)
  edge_index = data.edge_index.cpu().detach().numpy()
  rows = list(edge_index[0])
  cols = list(edge_index[1])
  for ind, (i, j) in enumerate(zip(rows, cols)):
    graph.add_edge(i, j)

  # Create the k-tuple labeled local graph
  tuple_graph = nx.Graph(directed=False)
  tuple_to_nodes = {} 
  nodes_to_tuple = {}

  for ind,tup in enumerate(list(itertools.product(list(graph.nodes),repeat=k))):
    # tuple_graph.add_node(ind,vector=node_attr) 
    tuple_to_nodes[tup] = ind
    nodes_to_tuple[ind] = tup

    # labels information
    label_info = np.concatenate([nx.get_node_attributes(graph, 'vector')[t] for t in tup])
    
    # structure information
    structure_info = tuple_structure(tup,graph)

    # Concatenate
    node_attr = np.concatenate(
        [label_info, structure_info], axis=-1)

    tuple_graph.add_node(ind,vector=node_attr)


  neighbors = {key: [] for key in range(k)}
  index = {key: [] for key in range(k)}
  edge_index = {key: [] for key in range(k)}

  # Convert node attributes to numpy arrays
  node_attrs = nx.get_node_attributes(tuple_graph, 'vector')
  node_features = np.array(list(node_attrs.values()))
  node_features = node_features.reshape((len(node_features), -1))

  for node in tuple_graph.nodes:

    # Get underlying nodes.
    tup = nodes_to_tuple[node]

    for i in range(k):
      index[i].append(int(tup[i]))
      for i_neighbor in graph.neighbors(tup[i]):
        i_neighbor_tuple = tup[:i]+tuple([i_neighbor])+tup[i+1:] # TODO speed up by creating a copy and replacing i-th component
        s = tuple_to_nodes[i_neighbor_tuple]
        neighbors[i].append([int(node),int(s)])
        tuple_graph.add_edge(int(node),int(s),label=f"{i}-local")

      edge_index[i] = torch.tensor(neighbors[i]).t().contiguous()


  # Convert back to pytorch geometric data format
  data_new = Data()
  data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

  # use for loop to assign edge_indices
  for i in range(k):
    data_new[f"edge_index_{i}"] = edge_index[i]
    data_new[f"index_{i}"] = index[i]

  data_new.y = data.y

  return data_new


def paral_local_transformer(dataset,k=2,n_jobs=4):
    results = Parallel(n_jobs=n_jobs)(delayed(tuple_k_local_transformer_iso)(data,k=k) for data in dataset)
    return results