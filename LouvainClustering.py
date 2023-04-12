import pandas as pd
import rdflib
from rdflib import Graph, OWL
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
import networkx as nx
import matplotlib.pyplot as plt
import networkx.algorithms.community as nx_comm


#### create NetworkX graph ####

# parse graph
g = Graph()
g.parse("./data/KG.ttl")

# turn graph into networkx graph
nx_graph = rdflib_to_networkx_digraph(g)
print("Number of Nodes: {n}".format(n=nx.number_of_nodes(nx_graph)))
print("Number of Edges: {n}".format(n=nx.number_of_edges(nx_graph)))

# rename the columns without the urls for readability
mapping = pd.DataFrame(nx_graph.nodes())
mapping['new_names'] = mapping[0].str.split("#",n=1,expand=False)
mapping['label'] = 'NA'
mapping_copy = mapping.copy()

for ind, m in mapping_copy.iterrows():
    l = len(m['new_names'])
    names = m['new_names']
    mapping.loc[ind,'label'] = names[l-1]
   
map_dict = dict(zip(mapping[0],mapping['label']))
nx_graph_nl = nx.relabel_nodes(nx_graph,map_dict,copy=True)

# find owl classes
EX = rdflib.Namespace("http://test.org/myonto.owl#")
g.bind("ex", EX)
owl_classes = []
for s,p,o in g.triples((None, None, OWL.Class)):
    owl_classes.append(s.split("#")[1])

# find edges
subjects = []
for s,p,o in g.triples((None, EX.hasAge, None)):
    subjects.append(s.split("#")[1])
for s,p,o in g.triples((None, EX.occuredIn, None)):
    subjects.append(s.split("#")[1])

edges = []
for edge in nx.edges(nx_graph_nl):
    if edge[0] in subjects and edge[1] != 'nan':
        edges.append(edge)


#### clustering ####

# find communities
communities = nx_comm.louvain_communities(nx_graph_nl,resolution=1) #resolution: high number favor small communitites, low favor large communities
print('Number of found communitites', len(communities),
'\n Number of nodes in the graph',nx.number_of_nodes(nx_graph_nl))

for i in range(len(communities)):
    print(f'{i}th community: \n',communities[i] )
    

# draw whole graph
pos = nx.shell_layout(nx_graph_nl)
nx.draw(nx_graph_nl, pos, edge_color='k',font_weight='light', 
        node_size= 100, width= 0.8)
plt.show()

# draw communities
for com in communities:
    nx.draw_networkx_nodes(nx_graph_nl,
                           pos,
                           nodelist=com, 
                           node_color = range(len(com)),
                           label=True,
                           node_size=100)  
    plt.show()

