import pandas as pd
import rdflib
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from rdflib.namespace import DC, FOAF

import networkx as nx
from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty

g = Graph()
g.parse("./data/Project")
EX = rdflib.Namespace("http://test.org/myonto.owl#")
g.bind("ex", EX)
t = 0
test = URIRef(EX+str(10016296))
for s,p,o in g.triples((test, None, None)):
    if o != RDFS.Resource:
        print(s,p,o)
        t+=1
    if t>200:
        break
        