import torch
from torch_geometric.data import (InMemoryDataset, Data)
import time
import numpy as np
import networkx as nx
import itertools
from tqdm import tqdm

def set_2_3_local_nonlocal_transformer(dataset, k, variant='local',star_variant = False):

  data_list = []
  N = dataset[0].x.shape[1]
  
  if star_variant:
    N+=1
  
  mapping_dict = {}

  # function that takes as an argument a set and the graph and return the hash for the set as above
  def assign_attr(Set,graph):
    dic = {l: [] for l in range(N)} 
    subgraph = graph.subgraph(Set)
    for node in subgraph.nodes:
      i = int(subgraph.nodes[node]['vector'])
      dic[i].append(subgraph.degree(node))
      dic[i]=sorted(dic[i])
    my_list=list(dic.values())
    joined = '|'.join([''.join(map(str, sublst)) for sublst in my_list])
    return joined

  for data in tqdm(dataset):
    if data.x.shape[0]<=k:
      continue
    data.edge_attr = None
    graph = nx.Graph(directed=False)

    if star_variant:
      for i in range(data.x.shape[0]):
        node_attr = np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)
      star_attr = max([graph.nodes[i]['vector'] for i in range(len(graph.nodes))])+1
      graph.add_node(i+1, vector=star_attr)

    if star_variant==False:
      # Add nodes with their attributes
      for i in range(data.x.shape[0]):
        node_attr = np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)

    set_graph = nx.Graph(directed=False)
    set_to_nodes = {} 
    nodes_to_set = {}
    
    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      set_to_nodes[tuple(sorted_set)] = ind
      nodes_to_set[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)
    
      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      set_graph.add_node(ind,attr=mapping_dict[node_attr])
      
    neighbors = {key: [] for key in range(k)}
    all_neighbors = []
    index = []

    for node in set_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_set[node])) # maybe ordering not necessary

      # check it again
      for i in range(k):
        index.append(tup[i])

        if variant =='local':
          # neighbors.
          neighbors[i] = list(graph.neighbors(tup[i]))

        if variant == 'nonlocal':
          neighbors[i] = list(graph.nodes)

        other_vertices = set(tup[:i]+tup[i+1:])
        if len((other_vertices).intersection(set(neighbors[i]))) != 0:
          neighbors[i] = [x for x in neighbors[i] if x not in list(other_vertices)]

        for neighbor in neighbors[i]: 
          tuple_neighbor = tup[:i]+tuple([neighbor])+tup[i+1:]
          s = set_to_nodes[tuple(sorted(tuple_neighbor))]
          all_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index = torch.tensor(all_neighbors).t().contiguous()

    data_new.edge_index = edge_index

    node_attrs = nx.get_node_attributes(set_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)

  return data_list

#Delta variant
def set_2_3_delta_transformer(dataset,k):

  # function that takes as an argument a set and the graph and return the hash for the set as above
  def assign_attr(Set,graph):
    dic = {l: [] for l in range(N)} 
    subgraph = graph.subgraph(Set)
    for node in subgraph.nodes:
      i = int(subgraph.nodes[node]['vector'])
      dic[i].append(subgraph.degree(node))
      dic[i]=sorted(dic[i])
    my_list=list(dic.values())
    joined = '|'.join([''.join(map(str, sublst)) for sublst in my_list])
    return joined

  N = dataset[0].x.shape[1]
  s = time.time()
  data_list = []
  mapping_dict = {}

  for data in tqdm(dataset):
    if data.x.shape[0]<=k:
      continue

    data.edge_attr = None
    graph = nx.Graph(directed=False)
    # Add nodes with their attributes
    for i in range(data.x.shape[0]):
      node_attr = np.argmax(data.x[i,:].cpu().detach().numpy())
      graph.add_node(i,vector=node_attr) # it is not a vector but a number indicating the position of "1" in one hot encoding vector. e.g. [0,0,0,1,0,0]-->3 (starting counting from 0)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)

    set_graph = nx.Graph(directed=False)
    set_to_nodes = {} 
    nodes_to_set = {}

    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      set_to_nodes[tuple(sorted_set)] = ind
      nodes_to_set[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      set_graph.add_node(ind,attr=mapping_dict[node_attr])
      
    # neighbors = {key: [] for key in range(k)}
    # all_neighbors = []
    index = []
    global_neighbors = {key: [] for key in range(k)}
    all_global_neighbors = []
    local_neighbors = {key: [] for key in range(k)}
    all_local_neighbors = []

    for node in set_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_set[node])) # maybe ordering not necessary

      # check it again
      for i in range(k):
        index.append(tup[i])

        # local neighbors.
        local_neighbors[i] = list(graph.neighbors(tup[i]))

        other_vertices = set(tup[:i]+tup[i+1:])
        if len((other_vertices).intersection(set(local_neighbors[i]))) != 0:
          local_neighbors[i] = [x for x in local_neighbors[i] if x not in list(other_vertices)]

        for local_neighbor in local_neighbors[i]: 
          tuple_local_neighbor = tup[:i]+tuple([local_neighbor])+tup[i+1:]
          s = set_to_nodes[tuple(sorted(tuple_local_neighbor))]
          all_local_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary

        # global neighbors.
        global_neighbors[i] = list(set(graph.nodes)-set(graph.neighbors(tup[i])))

        other_vertices = set(tup[:i]+tup[i+1:])
        if len((other_vertices).intersection(set(global_neighbors[i]))) != 0:
          global_neighbors[i] = [x for x in global_neighbors[i] if x not in list(other_vertices)]

        for global_neighbor in global_neighbors[i]: 
          tuple_global_neighbor = tup[:i]+tuple([global_neighbor])+tup[i+1:]
          s = set_to_nodes[tuple(sorted(tuple_global_neighbor))]
          all_global_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index_local = torch.tensor(all_local_neighbors).t().contiguous()
    edge_index_global = torch.tensor(all_global_neighbors).t().contiguous()

    data_new.edge_index_local = edge_index_local
    data_new.edge_index_global = edge_index_global


    node_attrs = nx.get_node_attributes(set_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)

  return data_list


def multiset_2_3_local_nonlocal_transformer(dataset,k,variant='local',star_variant = False):
  data_list = [] 
  N = dataset[0].x.shape[1] #number of node features in original graphs
  if star_variant:
    N+=1

  mapping_dict = {}

  # function that takes as an argument a multiset and the graph and return the hash for the isomorphism type
  def assign_attr(multiset,graph):
    dic = {l: [] for l in range(N)} 
    subgraph = graph.subgraph(multiset)
    distinct_counter = np.zeros(N)
    for node in subgraph.nodes:
      copies = multiset.count(node)
      i = int(subgraph.nodes[node]['vector'])
      for _ in range(copies):
        dic[i].append(subgraph.degree(node))
      dic[i]=sorted(dic[i])
      distinct_counter[i]+=1
    my_list=list(dic.values())
    joined = '|'.join('{}-{}'.format(''.join(map(str, sublst)), str(int(distinct_counter[i]))) for i, sublst in enumerate(my_list))
    return joined

  for data in tqdm(dataset):
    data.edge_attr = None
    graph = nx.Graph(directed=False)

    if star_variant:
      for i in range(data.x.shape[0]):
        node_attr = np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)
      star_attr = max([graph.nodes[i]['vector'] for i in range(len(graph.nodes))])+1
      graph.add_node(i+1, vector=star_attr)

    if star_variant==False:
      # Add nodes with their attributes
      for i in range(data.x.shape[0]):
        node_attr = np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)
    # nx.get_node_attributes(graph,'vector')

    # Add nodes to each set_graph
    multiset_graph = nx.Graph(directed=False)
    multiset_to_nodes = {} 
    nodes_to_multiset = {}

    for ind,multiset in enumerate(list(itertools.combinations_with_replacement(graph.nodes, k))):
      sorted_multiset = sorted(multiset)
      multiset_to_nodes[tuple(sorted_multiset)] = ind
      nodes_to_multiset[ind] = tuple(sorted_multiset)
      node_attr =  assign_attr(multiset,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      multiset_to_nodes[tuple(sorted_set)] = ind
      nodes_to_multiset[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    # Add edges to each set_graph

    # neighbors = {key: [] for key in range(k)} we dont have k distict elements 

    all_neighbors = []
    index = []

    for node in multiset_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_multiset[node])) # maybe ordering not necessary
      num_distinct = len(set(tup))
      neighbors = {key: [] for key in range(num_distinct)}
      distinct_elements = tuple(sorted(list(set(tup))))

      # check it again for node based tasks
      for i in range(num_distinct):
        index.append(distinct_elements[i])

        # neighbors by replacing the i-th element from the tuple of all distinct elements
        if variant == 'local':
          neighbors[i] = list(graph.neighbors(distinct_elements[i]))
        if variant == 'nonlocal':
          neighbors[i] = list(graph.nodes)

        mult_position = tup.index(distinct_elements[i]) # the position (one of them) where is the specific element in the multiset

        for neighbor in neighbors[i]: 
          # tuple_neighbor = tup[:mult_position]+tuple([neighbor])+tup[mult_position+1:]
          list_item = list(tup)
          list_item[mult_position] = neighbor
          tuple_neighbor = tuple(list_item)
          s = multiset_to_nodes[tuple(sorted(tuple_neighbor))]
          all_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index = torch.tensor(all_neighbors).t().contiguous()

    data_new.edge_index = edge_index

    node_attrs = nx.get_node_attributes(multiset_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)
  
  return data_list

def multiset_2_3_delta_transformer(dataset,k):
  data_list = [] 
  N = dataset[0].x.shape[1] #number of node features in original graphs
  mapping_dict = {}

  # function that takes as an argument a multiset and the graph and return the hash for the isomorphism type
  def assign_attr(multiset,graph):
    dic = {l: [] for l in range(N)} 
    subgraph = graph.subgraph(multiset)
    distinct_counter = np.zeros(N)
    for node in subgraph.nodes:
      copies = multiset.count(node)
      i = int(subgraph.nodes[node]['vector'])
      for _ in range(copies):
        dic[i].append(subgraph.degree(node))
      dic[i]=sorted(dic[i])
      distinct_counter[i]+=1
    my_list=list(dic.values())
    joined = '|'.join('{}-{}'.format(''.join(map(str, sublst)), str(int(distinct_counter[i]))) for i, sublst in enumerate(my_list))
    return joined

  for data in tqdm(dataset):
    data.edge_attr = None
    graph = nx.Graph(directed=False)

    # Add nodes with their attributes
    for i in range(data.x.shape[0]):
      node_attr = np.argmax(data.x[i,:].cpu().detach().numpy())
      graph.add_node(i,vector=node_attr)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)
    # nx.get_node_attributes(graph,'vector')

    # Add nodes to each set_graph
    multiset_graph = nx.Graph(directed=False)
    multiset_to_nodes = {} 
    nodes_to_multiset = {}

    for ind,multiset in enumerate(list(itertools.combinations_with_replacement(graph.nodes, k))):
      sorted_multiset = sorted(multiset)
      multiset_to_nodes[tuple(sorted_multiset)] = ind
      nodes_to_multiset[ind] = tuple(sorted_multiset)
      node_attr =  assign_attr(multiset,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      multiset_to_nodes[tuple(sorted_set)] = ind
      nodes_to_multiset[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    # Add edges to each set_graph

    # neighbors = {key: [] for key in range(k)} we dont have k distict elements 

    all_neighbors_local = []
    all_neighbors_global = []
    index = []

    for node in multiset_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_multiset[node])) # maybe ordering not necessary
      num_distinct = len(set(tup))
      neighbors_local = {key: [] for key in range(num_distinct)}
      neighbors_global = {key: [] for key in range(num_distinct)}
      distinct_elements = tuple(sorted(list(set(tup))))

      # check it again for node based tasks
      for i in range(num_distinct):
        index.append(distinct_elements[i])

        neighbors_local[i] = list(graph.neighbors(distinct_elements[i]))
        neighbors_global[i] = list(set(graph.nodes)-set(graph.neighbors(distinct_elements[i])))
        mult_position = tup.index(distinct_elements[i]) # the position (one of them) where is the element in the multiset (if in many positions returns the first)

        for local_neighbor in neighbors_local[i]: 
          # local_tuple_neighbor = tup[:mult_position]+tuple([local_neighbor])+tup[mult_position+1:]
          list_item = list(tup)
          list_item[mult_position] = local_neighbor
          local_tuple_neighbor = tuple(list_item)
          s = multiset_to_nodes[tuple(sorted(local_tuple_neighbor))]
          all_neighbors_local.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s),label='local')) #not necessary
        
        
        for global_neighbor in neighbors_global[i]: 
          # global_tuple_neighbor = tup[:mult_position]+tuple([global_neighbor])+tup[mult_position+1:]
          list_item = list(tup)
          list_item[mult_position] = global_neighbor
          global_tuple_neighbor = tuple(list_item)
          s = multiset_to_nodes[tuple(sorted(global_tuple_neighbor))]
          all_neighbors_global.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s),label='global')) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index_local = torch.tensor(all_neighbors_local).t().contiguous()
    data_new.edge_index_local = edge_index_local

    edge_index_global = torch.tensor(all_neighbors_global).t().contiguous()
    data_new.edge_index_global = edge_index_global

    node_attrs = nx.get_node_attributes(multiset_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)
  
  return data_list

# Convert to k-tuple local graph
# k=3 ---> 3h 45m
def tuple_k_local_transformer(dataset,k):
  
  data_list = []

  # Given a graph in networkx format and a k-tuple returns the kxk matrix M with entries {0,1,2} (as discussed) (actually returns the uppertriangular part flattened)
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

  for data in tqdm(dataset):
    data.edge_attr = None 

    # Create the graph
    graph = nx.Graph(directed=False)

    # Add nodes with their attributes
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

    data_list.append(data_new)

  return data_list

# Convert to 2-tuple local graph
def tuple_2_local_transformer(dataset):

  data_list=[]
  for data in tqdm(dataset):
    # Drop edge attributes 
    data.edge_attr = None 

    # Create the graph
    graph = nx.Graph(directed=False)

    # Add nodes with their attributes
    for i in range(data.x.shape[0]):
      node_attr = data.x[i,:].cpu().detach().numpy()
      graph.add_node(i,vector=node_attr)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)

    # Create the 2-tuple labeled local graph
    tuple_graph = nx.Graph(directed=False)
    # type = {}
    tuple_to_nodes = {} 
    nodes_to_tuple = {}

    for ind,tup in enumerate(list(itertools.product(list(graph.nodes),repeat=2))):
      # tuple_graph.add_node(ind,vector=node_attr) 
      tuple_to_nodes[tup] = ind
      nodes_to_tuple[ind] = tup

      if tup[0]==tup[1]:
        node_feature0 = nx.get_node_attributes(graph,'vector')[tup[0]]
        node_feature1 = nx.get_node_attributes(graph,'vector')[tup[1]]
        node_attr = np.concatenate(
            [node_feature0, node_feature1, np.array([1,0])], axis=-1) # 1 indicates repeatition of coordinates and 0 indicates there is not edge (i.e no self loop)
        tuple_graph.add_node(ind,vector=node_attr)

      else:
        node_feature0 = nx.get_node_attributes(graph,'vector')[tup[0]]
        node_feature1 = nx.get_node_attributes(graph,'vector')[tup[1]]
        if ((tup[0],tup[1]) in graph.edges) or ((tup[1],tup[0]) in graph.edges):
          node_attr = np.concatenate(
            [node_feature0, node_feature1, np.array([0,1])], axis=-1) # 0 indicates no repeatition of coordinates and 1 indicates there is an edge between coordinates
          tuple_graph.add_node(ind,vector=node_attr)
        else:
          node_attr = np.concatenate(
            [node_feature0, node_feature1, np.array([0,0])], axis=-1) # 0 indicates no repeatition of coordinates and 0 indicates there is not an edge between coordinates
          tuple_graph.add_node(ind,vector=node_attr)

    neighbors_0 = [] # for the pytorch geometric type data
    neighbors_1 = [] # for the pytorch geometric type data

    # Convert node attributes to numpy arrays
    node_attrs = nx.get_node_attributes(tuple_graph, 'vector')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))

    index_0 = [] # for the pytorch geometric type data
    index_1 = [] # for the pytorch geometric type data

    for node in tuple_graph.nodes:
        # Get underlying nodes.
        v, w = nodes_to_tuple[node]

        index_0.append(int(v))
        index_1.append(int(w))

        # 1 neighbors.
        for neighbor in graph.neighbors(v): # We consider only 1-local neighbors
          s = tuple_to_nodes[(neighbor, w)]
          neighbors_0.append([int(node), int(s)])
          # tuple_graph.add_edge(int(node),int(s),label='1-Local') #not necessary

        # 2 neighbors.
        for neighbor in graph.neighbors(w):
          s = tuple_to_nodes[(v, neighbor)]
          neighbors_1.append([int(node), int(s)])
          # tuple_graph.add_edge(int(node),int(s),label='2-Local') # not necessary

    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index_0 = torch.tensor(neighbors_0).t().contiguous()
    edge_index_1 = torch.tensor(neighbors_1).t().contiguous()

    data_new.edge_index_0 = edge_index_0
    data_new.edge_index_1 = edge_index_1

    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)
    data_new.index_0 = torch.from_numpy(np.array(index_0)).to(torch.int64) #where are they used?
    data_new.index_1 = torch.from_numpy(np.array(index_1)).to(torch.int64) #where are they used?

    data_new.y = data.y

    data_list.append(data_new)

  return data_list

# for graphs with not initial node features (nif)
def set_2_3_local_nonlocal_transformer_nif(dataset, k, variant='local',star_variant = False):

  data_list = []  
  mapping_dict = {}

  np.random.seed(42)
  fixed_node_attr = np.random.uniform(low=0.0, high=1.0, size=3)
  np.random.seed(7)
  fixed_node_attr_star = np.random.uniform(low=0.0, high=1.0, size=3)

  def assign_attr(Set,graph):
    subgraph = graph.subgraph(Set)
    return len(subgraph.edges)
  
  for data in tqdm(dataset):
    data.edge_attr = None
    graph = nx.Graph(directed=False)

    if star_variant:
      for i in range(data.num_nodes):
        node_attr = fixed_node_attr
        graph.add_node(i,vector=node_attr)
      graph.add_node(i+1, vector=fixed_node_attr_star)

    if star_variant==False:
      # Add nodes with their attributes
      for i in range(data.num_nodes):
        node_attr = np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)

    set_graph = nx.Graph(directed=False)
    set_to_nodes = {} 
    nodes_to_set = {}
    
    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      set_to_nodes[tuple(sorted_set)] = ind
      nodes_to_set[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)
    
      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      set_graph.add_node(ind,attr=mapping_dict[node_attr])
      

    
    neighbors = {key: [] for key in range(k)}
    all_neighbors = []
    index = []

    for node in set_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_set[node])) # maybe ordering not necessary

      # check it again
      for i in range(k):
        index.append(tup[i])

        if variant =='local':
          # neighbors.
          neighbors[i] = list(graph.neighbors(tup[i]))

        if variant == 'nonlocal':
          neighbors[i] = list(graph.nodes)

        other_vertices = set(tup[:i]+tup[i+1:])
        if len((other_vertices).intersection(set(neighbors[i]))) != 0:
          neighbors[i] = [x for x in neighbors[i] if x not in list(other_vertices)]

        for neighbor in neighbors[i]: 
          tuple_neighbor = tup[:i]+tuple([neighbor])+tup[i+1:]
          s = set_to_nodes[tuple(sorted(tuple_neighbor))]
          all_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index = torch.tensor(all_neighbors).t().contiguous()

    data_new.edge_index = edge_index

    node_attrs = nx.get_node_attributes(set_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)

  return data_list


# for graphs with not initial node features (nif)
def multiset_2_3_local_nonlocal_transformer_nif(dataset,k,variant='local',star_variant = False):
  
  data_list = [] 
  mapping_dict = {}

  np.random.seed(42)
  fixed_node_attr = np.random.uniform(low=0.0, high=1.0, size=3)
  np.random.seed(7)
  fixed_node_attr_star = np.random.uniform(low=0.0, high=1.0, size=3)

  # function that takes as an argument a multiset and the graph and return the hash for the isomorphism type
  def assign_attr(Set,graph):
    subgraph = graph.subgraph(Set)
    return len(subgraph.edges)
  
  for data in tqdm(dataset):
    data.edge_attr = None
    graph = nx.Graph(directed=False)

    if star_variant:
      for i in range(data.num_nodes):
        node_attr = fixed_node_attr#np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)
      graph.add_node(i+1, vector=fixed_node_attr_star)

    if star_variant==False:
      # Add nodes with their attributes
      for i in range(data.num_nodes):
        node_attr = fixed_node_attr #np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)
    # nx.get_node_attributes(graph,'vector')

    # Add nodes to each set_graph
    multiset_graph = nx.Graph(directed=False)
    multiset_to_nodes = {} 
    nodes_to_multiset = {}

    for ind,multiset in enumerate(list(itertools.combinations_with_replacement(graph.nodes, k))):
      sorted_multiset = sorted(multiset)
      multiset_to_nodes[tuple(sorted_multiset)] = ind
      nodes_to_multiset[ind] = tuple(sorted_multiset)
      node_attr =  assign_attr(multiset,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      multiset_to_nodes[tuple(sorted_set)] = ind
      nodes_to_multiset[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    # Add edges to each set_graph

    # neighbors = {key: [] for key in range(k)} we dont have k distict elements 

    all_neighbors = []
    index = []

    for node in multiset_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_multiset[node])) # maybe ordering not necessary
      num_distinct = len(set(tup))
      neighbors = {key: [] for key in range(num_distinct)}
      distinct_elements = tuple(sorted(list(set(tup))))

      # check it again for node based tasks
      for i in range(num_distinct):
        index.append(distinct_elements[i])

        # neighbors by replacing the i-th element from the tuple of all distinct elements
        if variant == 'local':
          neighbors[i] = list(graph.neighbors(distinct_elements[i]))
        if variant == 'nonlocal':
          neighbors[i] = list(graph.nodes)

        mult_position = tup.index(distinct_elements[i]) # the position (one of them) where is the specific element in the multiset

        for neighbor in neighbors[i]: 
          # tuple_neighbor = tup[:mult_position]+tuple([neighbor])+tup[mult_position+1:]
          list_item = list(tup)
          list_item[mult_position] = neighbor
          tuple_neighbor = tuple(list_item)
          s = multiset_to_nodes[tuple(sorted(tuple_neighbor))]
          all_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index = torch.tensor(all_neighbors).t().contiguous()

    data_new.edge_index = edge_index

    node_attrs = nx.get_node_attributes(multiset_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)
  
  return data_list

# for graphs with not initial node features (nif)
def set_2_3_delta_transformer_nif(dataset,k):

  # function that takes as an argument a set and the graph and return the hash for the set as above
  def assign_attr(Set,graph):
    subgraph = graph.subgraph(Set)
    return len(subgraph.edges)

  # N = dataset[0].x.shape[1]
  s = time.time()
  data_list = []
  mapping_dict = {}
  np.random.seed(42)
  fixed_node_attr = np.random.uniform(low=0.0, high=1.0, size=3)
  
  for data in tqdm(dataset):

    data.edge_attr = None
    graph = nx.Graph(directed=False)
    # Add nodes with their attributes
    for i in range(data.num_nodes):
      node_attr = fixed_node_attr#np.argmax(data.x[i,:].cpu().detach().numpy())
      graph.add_node(i,vector=node_attr) # it is not a vector but a number indicating the position of "1" in one hot encoding vector. e.g. [0,0,0,1,0,0]-->3 (starting counting from 0)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)

    set_graph = nx.Graph(directed=False)
    set_to_nodes = {} 
    nodes_to_set = {}

    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      set_to_nodes[tuple(sorted_set)] = ind
      nodes_to_set[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      set_graph.add_node(ind,attr=mapping_dict[node_attr])
      


    # neighbors = {key: [] for key in range(k)}
    # all_neighbors = []
    index = []
    global_neighbors = {key: [] for key in range(k)}
    all_global_neighbors = []
    local_neighbors = {key: [] for key in range(k)}
    all_local_neighbors = []

    for node in set_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_set[node])) # maybe ordering not necessary

      # check it again
      for i in range(k):
        index.append(tup[i])

        # local neighbors.
        local_neighbors[i] = list(graph.neighbors(tup[i]))

        other_vertices = set(tup[:i]+tup[i+1:])
        if len((other_vertices).intersection(set(local_neighbors[i]))) != 0:
          local_neighbors[i] = [x for x in local_neighbors[i] if x not in list(other_vertices)]

        for local_neighbor in local_neighbors[i]: 
          tuple_local_neighbor = tup[:i]+tuple([local_neighbor])+tup[i+1:]
          s = set_to_nodes[tuple(sorted(tuple_local_neighbor))]
          all_local_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary

        # global neighbors.
        global_neighbors[i] = list(set(graph.nodes)-set(graph.neighbors(tup[i])))

        other_vertices = set(tup[:i]+tup[i+1:])
        if len((other_vertices).intersection(set(global_neighbors[i]))) != 0:
          global_neighbors[i] = [x for x in global_neighbors[i] if x not in list(other_vertices)]

        for global_neighbor in global_neighbors[i]: 
          tuple_global_neighbor = tup[:i]+tuple([global_neighbor])+tup[i+1:]
          s = set_to_nodes[tuple(sorted(tuple_global_neighbor))]
          all_global_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index_local = torch.tensor(all_local_neighbors).t().contiguous()
    edge_index_global = torch.tensor(all_global_neighbors).t().contiguous()

    data_new.edge_index_local = edge_index_local
    data_new.edge_index_global = edge_index_global


    node_attrs = nx.get_node_attributes(set_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)

  return data_list

#for graphs without initial node features (nif)
def multiset_2_3_local_transformer(dataset,k,variant='local',star_variant = False):
  data_list = [] 
  mapping_dict = {}

  # function that takes as an argument a multiset and the graph and return the hash for the isomorphism type
  def assign_attr(Set,graph):
    subgraph = graph.subgraph(Set)
    return len(subgraph.edges)

  np.random.seed(42)
  fixed_node_attr = np.random.uniform(low=0.0, high=1.0, size=3)
  
  for data in tqdm(dataset):
    data.edge_attr = None
    graph = nx.Graph(directed=False)

    if star_variant:
      for i in range(data.num_nodes):
        node_attr = fixed_node_attr#np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)
      star_attr = max([graph.nodes[i]['vector'] for i in range(len(graph.nodes))])+1
      graph.add_node(i+1, vector=star_attr)

    if star_variant==False:
      # Add nodes with their attributes
      for i in range(data.num_nodes):
        node_attr = fixed_node_attr #np.argmax(data.x[i,:].cpu().detach().numpy())
        graph.add_node(i,vector=node_attr)

    # Add edges (without their attributes)
    edge_index = data.edge_index.cpu().detach().numpy()
    rows = list(edge_index[0])
    cols = list(edge_index[1])
    for ind, (i, j) in enumerate(zip(rows, cols)):
      graph.add_edge(i, j)
    # nx.get_node_attributes(graph,'vector')

    # Add nodes to each set_graph
    multiset_graph = nx.Graph(directed=False)
    multiset_to_nodes = {} 
    nodes_to_multiset = {}

    for ind,multiset in enumerate(list(itertools.combinations_with_replacement(graph.nodes, k))):
      sorted_multiset = sorted(multiset)
      multiset_to_nodes[tuple(sorted_multiset)] = ind
      nodes_to_multiset[ind] = tuple(sorted_multiset)
      node_attr =  assign_attr(multiset,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      multiset_to_nodes[tuple(sorted_set)] = ind
      nodes_to_multiset[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)

      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) #np.array([len(mapping_dict)])

      multiset_graph.add_node(ind,attr=mapping_dict[node_attr])

    # Add edges to each set_graph

    # neighbors = {key: [] for key in range(k)} we dont have k distict elements 

    all_neighbors = []
    index = []

    for node in multiset_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_multiset[node])) # maybe ordering not necessary
      num_distinct = len(set(tup))
      neighbors = {key: [] for key in range(num_distinct)}
      distinct_elements = tuple(sorted(list(set(tup))))

      # check it again for node based tasks
      for i in range(num_distinct):
        index.append(distinct_elements[i])

        # neighbors by replacing the i-th element from the tuple of all distinct elements
        if variant == 'local':
          neighbors[i] = list(graph.neighbors(distinct_elements[i]))
        if variant == 'nonlocal':
          neighbors[i] = list(graph.nodes)

        mult_position = tup.index(distinct_elements[i]) # the position (one of them) where is the specific element in the multiset

        for neighbor in neighbors[i]: 
          # tuple_neighbor = tup[:mult_position]+tuple([neighbor])+tup[mult_position+1:]
          list_item = list(tup)
          list_item[mult_position] = neighbor
          tuple_neighbor = tuple(list_item)
          s = multiset_to_nodes[tuple(sorted(tuple_neighbor))]
          all_neighbors.append([int(node), int(s)])
          # set_graph.add_edge(int(node),int(s)) #not necessary
            
    # Convert back to pytorch geometric data format
    data_new = Data()

    edge_index = torch.tensor(all_neighbors).t().contiguous()

    data_new.edge_index = edge_index

    node_attrs = nx.get_node_attributes(multiset_graph, 'attr')
    node_features = np.array(list(node_attrs.values()))
    node_features = node_features.reshape((len(node_features), -1))
    data_new.x = torch.from_numpy(np.array(node_features)).to(torch.float)

    # Check it again
    data_new.index = torch.from_numpy(np.array(index)).to(torch.int64) #where are they used? ---> for the scatter aggregation function

    data_new.y = data.y

    data_list.append(data_new)

  n = len(mapping_dict)

  # One hot encoding
  for d in data_list:
    vrtcs = d.x.shape[0]
    new_x = np.zeros((vrtcs, n))
    for i,j in enumerate(d.x):
      new_x[i, int(j)] = 1
    d.x = torch.tensor(new_x,dtype=torch.float32)
  
  return data_list