import networkx as nx
import itertools
import numpy as np


def partition(lst):
    value_indices = {}
    for index, value in enumerate(lst):
        if value not in value_indices:
            value_indices[value] = [index]
        else:
            value_indices[value].append(index)
    return list(value_indices.values())

def set_k(G1, G2, k, local = True , star_variant = False):
  graphs_list = []
  N = len(G1.nodes[0]['features'])

  if star_variant:
    N+=1

  mapping_dict = {}

  # function that takes as an argument a set and the graph and return the hash for the set
  def assign_attr(Set,graph):
    dic = {l: [] for l in range(N)} 
    subgraph = graph.subgraph(Set)
    for node in subgraph.nodes:
      i = int(np.argmax(np.array(subgraph.nodes[node]['features'])))
      dic[i].append(subgraph.degree(node))
      dic[i]=sorted(dic[i])

    my_list=list(dic.values())
    joined = '|'.join([''.join(map(str, sublst)) for sublst in my_list])
    return joined

  for graph in [G1,G2]:

    if star_variant:
      for i in range(len(graph.nodes())):
        graph.nodes[i]['features'] = graph.nodes[i]['features']+[0]

      graph.add_node(len(graph.nodes), features=[0]*(len(graph.nodes[0]['features'])-1) + [1] )

    set_graph = nx.Graph(directed=False)
    set_to_nodes = {} 
    nodes_to_set = {}
    
    for ind,Set in enumerate(list(itertools.combinations(graph.nodes, k))):
      sorted_set = sorted(Set)
      set_to_nodes[tuple(sorted_set)] = ind
      nodes_to_set[ind] = tuple(sorted_set)
      node_attr =  assign_attr(Set,graph)
    
      if node_attr not in mapping_dict.keys():
        mapping_dict[node_attr] = len(mapping_dict) 

      set_graph.add_node(ind,features=[mapping_dict[node_attr]])
      
    neighbors = {key: [] for key in range(k)}
    all_neighbors = []
    # index = []

    for node in set_graph.nodes:
      # Get underlying nodes.
      tup = tuple(sorted(nodes_to_set[node])) # maybe ordering not necessary

      # check it again
      for i in range(k):
        # index.append(tup[i])

        if local:
          # neighbors.
          neighbors[i] = list(graph.neighbors(tup[i]))

        if not local:
          neighbors[i] = list(graph.nodes)

        other_vertices = set(tup[:i]+tup[i+1:])
        if len((other_vertices).intersection(set(neighbors[i]))) != 0:
          neighbors[i] = [x for x in neighbors[i] if x not in list(other_vertices)]

        for neighbor in neighbors[i]: 
          tuple_neighbor = tup[:i]+tuple([neighbor])+tup[i+1:]
          s = set_to_nodes[tuple(sorted(tuple_neighbor))]
          all_neighbors.append([int(node), int(s)])
          set_graph.add_edge(int(node),int(s)) 

    graphs_list.append(set_graph)

  return graphs_list

def multiset_k(G1, G2, k, local = True, star_variant = False):
    
    graphs_list = [] 
    N = len(G1.nodes[0]['features'])

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
            i = int(np.argmax(np.array(subgraph.nodes[node]['features'])))
            for _ in range(copies):
                dic[i].append(subgraph.degree(node))
            dic[i]=sorted(dic[i])
            distinct_counter[i]+=1
        my_list=list(dic.values())
        joined = '|'.join('{}-{}'.format(''.join(map(str, sublst)), str(int(distinct_counter[i]))) for i, sublst in enumerate(my_list))
        return joined

    for graph in [G1,G2]:

        if star_variant:
            for i in range(len(graph.nodes())):
                graph.nodes[i]['features'] = graph.nodes[i]['features']+[0]

            graph.add_node(len(graph.nodes), features=[0]*(len(graph.nodes[0]['features'])-1) + [1] )


        multiset_graph = nx.Graph(directed=False)
        multiset_to_nodes = {} 
        nodes_to_multiset = {}

        for ind,multiset in enumerate(list(itertools.combinations_with_replacement(graph.nodes, k))):
            sorted_multiset = sorted(multiset)
            multiset_to_nodes[tuple(sorted_multiset)] = ind
            nodes_to_multiset[ind] = tuple(sorted_multiset)
            node_attr =  assign_attr(multiset,graph)

            if node_attr not in mapping_dict.keys():
                mapping_dict[node_attr] = len(mapping_dict) 

            multiset_graph.add_node(ind,features=[mapping_dict[node_attr]])

        all_neighbors = []

        for node in multiset_graph.nodes:
            # Get underlying nodes.
            tup = tuple(sorted(nodes_to_multiset[node])) # maybe ordering not necessary
            num_distinct = len(set(tup))
            neighbors = {key: [] for key in range(num_distinct)}
            distinct_elements = tuple(sorted(list(set(tup))))

            # check it again for node based tasks
            for i in range(num_distinct):

                # neighbors by replacing the i-th element from the tuple of all distinct elements
                if local:
                    neighbors[i] = list(graph.neighbors(distinct_elements[i]))
                if not local:
                    neighbors[i] = list(graph.nodes)

                mult_position = tup.index(distinct_elements[i]) # the position (one of them) where is the specific element in the multiset

                for neighbor in neighbors[i]: 
                    # tuple_neighbor = tup[:mult_position]+tuple([neighbor])+tup[mult_position+1:]
                    list_item = list(tup)
                    list_item[mult_position] = neighbor
                    tuple_neighbor = tuple(list_item)
                    s = multiset_to_nodes[tuple(sorted(tuple_neighbor))]
                    all_neighbors.append([int(node), int(s)])
                    multiset_graph.add_edge(int(node),int(s)) #not necessary
            

        graphs_list.append(multiset_graph)

    return graphs_list

def WL(G1, G2, variant = None, k = None, local = True, star = False):

    if variant == 'set':

        G1,G2 = set_k(G1, G2, k = k, local = local, star_variant = star)

    if variant == 'multiset':
        
        G1, G2 = multiset_k(G1, G2, k = k, local = local, star_variant = star)
        
    N = len(G1.nodes)
    h = 0
    old_features1 = [G1.nodes[i]['features'] for i in range(N)]
    old_features2 = [G2.nodes[i]['features'] for i in range(N)]
    termination = [False, None]
    hashing_labels = {}
    
    for attr in old_features1 + old_features2:
        if str(attr) in hashing_labels.keys():
            continue
        else:
            hashing_labels[str(attr)]=h
            h+=1
    old_features1 = [hashing_labels[str(_)] for _ in old_features1]
    old_features2 = [hashing_labels[str(_)] for _ in old_features2]
    c=0

    while termination[0] == False:
        print("Iteration:",c+1)
        for attr in old_features1 + old_features2:
            if str(attr) in hashing_labels.keys():
                continue
            else:
                hashing_labels[str(attr)]=h
                h+=1
        c+=1
        new_features1 = [None for _ in range(N)]
        new_features2 = [None for _ in range(N)]
        for i in range(N):
            neighborsG1_i = list(G1.neighbors(i))
            neighborsG2_i = list(G2.neighbors(i))
            neighbors_featuresG1 = sorted([G1.nodes[j]['features'] for j in neighborsG1_i])
            neighbors_featuresG2 = sorted([G2.nodes[j]['features'] for j in neighborsG2_i])
            new_features1[i] = [old_features1[i]] + ['-'] + neighbors_featuresG1
            new_features2[i] = [old_features2[i]] + ['-'] + neighbors_featuresG2
            if str(new_features1[i]) in hashing_labels.keys():
                new_features1[i] = hashing_labels[str(new_features1[i])]
            else:
                hashing_labels[str(new_features1[i])] = h
                new_features1[i] = h
                h+=1
            if str(new_features2[i]) in hashing_labels.keys():
                new_features2[i] = hashing_labels[str(new_features2[i])]
            else:
                hashing_labels[str(new_features2[i])] = h
                new_features2[i] = h
                h+=1

        for color in new_features1 + new_features2:
            if new_features1.count(color) != new_features2.count(color):
                termination = [True, "Non-isomorphic."]
                break
            else:
                if partition(old_features1)==partition(new_features1) and partition(old_features2)==partition(new_features2):
                    termination = [True, "Cannot determine."]
                    
        print(f"Old colors of G1:  {old_features1}")
        print(f"Old colors of G2:  {old_features2}")
        old_features1 = new_features1#[hashing_labels[str(_)] for _ in old_features1]
        old_features2 = new_features2#[hashing_labels[str(_)] for _ in old_features2]
        print(f"new colors of G1:  {old_features1}")
        print(f"new colors of G2:  {old_features2}")

    
    print("Decision:", termination[1],"Terminated in iteration:", c) 

    if termination[1] == "Non-isomorphic.":
       return True
    
    if termination[1] == "Cannot determine.":
       return False
            