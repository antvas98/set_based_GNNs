from itertools import chain, combinations
import networkx as nx

def CFI(k):
    # Changed for folklore WL graphs.
    # TODO
    feature = [0]*(k+2)#
    K = nx.complete_graph(k + 1)

    ## graph 1
    G = nx.Graph()

    # feat_e=0
    for i, e in enumerate(K.edges):
        ff0 = feature.copy()#
        ff0[0]=1
        G.add_node((e, 0), features = ff0 , data=str("e") + str(i))
        G.add_node((e, 1), features = ff0, data=str("e") + str(i))
        G.add_edge((e, 1), (e, 0))
    pos=0
    for u in K:
        pos+=1
        for S in subsetslist(0, K, u):
            ff = feature.copy()#
            ff[pos]=1
            G.add_node((u, S), features = ff, data=str(u))

            for e in incidentedges(K, u):
                G.add_edge((u, S), (e, int(e in S)))

    ## graph 2
    H = nx.Graph()

    # feat_e = 0
    for i, e in enumerate(K.edges):
        ff0=feature.copy()
        ff0[0]=1
        H.add_node((e, 0), features = ff0,data=str("e") + str(i))
        H.add_node((e, 1), features = ff0, data=str("e") + str(i))
        H.add_edge((e, 1), (e, 0))
    pos=0#
    for u in K:
        ff = feature.copy()
        pos+=1#  
        parity = int(u == 0)  ## vertex 0 in K, the "odd" one out
        for S in subsetslist(parity, K, u):
            ff[pos]=1#
            H.add_node((u, S), features = ff,data=str(u))
            for e in incidentedges(K, u):
                H.add_edge((u, S), (e, int(e in S)))

    G = nx.convert_node_labels_to_integers(G)
    H = nx.convert_node_labels_to_integers(H)

    return (G, H)

def incidentedges(K, u):
    return [tuple(sorted(e)) for e in K.edges(u)]


## generate all edge subsets of odd/even cardinality
## set parameter "parity" 0/1 for odd/even sets resp.

def subsetslist(parity, K, u):
    oddsets = set()
    evensets = set()
    for s in list(powerset(incidentedges(K, u))):
        if (len(s) % 2 == 0):
            evensets.add(frozenset(s))
        else:
            oddsets.add(frozenset(s))
    if parity == 0:
        return evensets
    else:
        return oddsets
    
## generate all subsets of a set
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))