import pykeen
from typing import List
import numpy as np
from pykeen import utils, triples, models
from sklearn.cluster import KMeans
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from pykeen.pipeline import pipeline
from pykeen.models import TransE
import torch.nn as nn
import torch
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
import rdflib
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from rdflib.namespace import DC, FOAF

g = Graph()
g.parse('./data/Project')

data = []
i=0
for s,p,o in g.triples((None,None,None)):
    if '#' in s and '#' in p and '#' in o:
        data.append([s.split("#")[1], p.split("#")[1], o.split("#")[1]])

t = np.array(data, dtype=str)

trip = triples.TriplesFactory.from_labeled_triples(t)

path_to_kg = ('./data/Project')
triples_factory = pykeen.triples.TriplesFactory.from_path(path=path_to_kg, create_inverse_triples=False)
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


# Cluster the embeddings using k-means
cluster_list = [2,3,4,5,6,7,8,9,10]
for i in cluster_list:
    num_clusters = i  # Set the number of clusters
    kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(entity_embedding_tensor)
    cluster_labels = kmeans.fit_predict(entity_embedding_tensor)

# Plot the data with color-coded clusters
plt.scatter(entity_embedding_tensor[:,0], entity_embedding_tensor[:,1], c = cluster_labels)
plt.show()

# Print the cluster labels
print("Cluster labels: ", cluster_labels)

EX = rdflib.Namespace("http://test.org/myonto.owl#")

# import ampligraph
# from ampligraph.latent_features import ScoringBasedEmbeddingModel

# model = ScoringBasedEmbeddingModel(k=100,
#                                    eta=20,
#                                    scoring_type='ComplEx')

# model.fit(X_train,
#           batch_size=int(X_train.shape[0] / 50),
#           epochs=300,  # Number of training epochs
#           verbose=True  # Enable stdout messages
#           )

# ppl = []
# for s in g.subjects(RDF.type, EX.Person):
#     print(s)
#     if "ex:Person" in s:
#         ppl.append(s)
# print("THIS IS PEOPLE", ppl)


# ppl_embeddings = dict(zip(ppl, model.get_embeddings(ppl)))
# ppl_embeddings_array = np.array([i for i in ppl_embeddings.values()])
# embeddings_2d = PCA(n_components=2).fit_transform(ppl_embeddings_array)

# clustering_algorithm = KMeans(n_clusters=6, n_init=100, max_iter=500, random_state=0)
# clusters = find_clusters(ppl, model, clustering_algorithm, mode='e')