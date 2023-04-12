import pykeen
from typing import List
import numpy as np
from pykeen import utils, triples, models
from sklearn.cluster import KMeans
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from pykeen.pipeline import pipeline
from pykeen.models import TransE
import torch.nn as nn
import torch
import re
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
import rdflib
from rdflib.namespace import DC, FOAF

g = Graph()
g.parse('./data/KG.ttl')

data = []
i=0
for s,p,o in g.triples((None,None,None)):
    if '#' in s and '#' in p and '#' in o:
        data.append([s.split("#")[1], p.split("#")[1], o.split("#")[1]])

t = np.array(data, dtype=str)

trip = triples.TriplesFactory.from_labeled_triples(t)

path_to_kg = ('./data/KG.ttl')
X_train, X_test = trip.split([0.95,0.05])

embedding_model = pykeen.models.TransE(triples_factory = trip)

result = pipeline(
    random_seed = 0,
    model = embedding_model,
    training_kwargs = dict(num_epochs = 2),
    evaluation_kwargs = dict(),
    training=X_train,
    testing=X_test,)


model = result.model

entity_representation_modules: List['pykeen.nn.Representation'] = model.entity_representations
relation_representation_modules: List['pykeen.nn.Representation'] = model.relation_representations
entity_embeddings: pykeen.nn.Embedding = entity_representation_modules[0]
relation_embeddings: pykeen.nn.Embedding = relation_representation_modules[0]
entity_embedding_tensor: torch.FloatTensor = entity_embeddings()
relation_embedding_tensor: torch.FloatTensor = relation_embeddings()
entity_embedding_tensor: torch.FloatTensor = entity_embeddings(indices=None)
relation_embedding_tensor: torch.FloatTensor = relation_embeddings(indices=None)
entity_embedding_tensor = model.entity_representations[0](indices=None).detach().numpy()

num_clusters = 5
kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(entity_embedding_tensor)
cluster_labels = kmeans.fit_predict(entity_embedding_tensor)

plt.scatter(entity_embedding_tensor[:,0], entity_embedding_tensor[:,1], c = cluster_labels)
plt.show()


print("Cluster labels: ", cluster_labels)