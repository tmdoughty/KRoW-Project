# # load the knowledge graph
# # knowledge_graph = from_path(r'C:\Users\dough\Documents\GitHub\KRoW-Project\data\Project_onto.owl')
# knowledge_graph = pykeen.triples.utils.load_triples(r'C:\Users\dough\Documents\GitHub\KRoW-Project\data\Project_onto.owl')
# # pykeen.triples.utils.load_triples()
# # extract features

# KG = pykeen.triples.TriplesFactory(knowledge_graph, entity_to_id = None, relation_to_id = None)
# model = pykeen.models.TransE(triples_factory = KG)
# model.fit(knowledge_graph)
# features = model.get_all_entities()

# # train the KMeans model
# kmeans = KMeans(n_clusters=3, random_state=0)
# kmeans.fit(features)

# # evaluate the model
# silhouette_coefficient = sklearn.metrics.silhouette_score(features, kmeans.labels_)
# inertia = kmeans.inertia_

# print(f"Silhouette Coefficient: {silhouette_coefficient}")
# print(f"Inertia: {inertia}")

import pykeen
import numpy as np
from pykeen import utils, triples, models
from sklearn.cluster import KMeans
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from pykeen.pipeline import pipeline
from pykeen.models import TransE

g = Graph()
g.parse(r'C:\Users\dough\Documents\GitHub\KRoW-Project\data\Project')
# g.parse("./data/Project")
# from krw_ml.py
data = []
i=0
for s,p,o in g.triples((None,None,None)):
    if '#' in s and '#' in p and '#' in o:
        data.append([s.split("#")[1], p.split("#")[1], o.split("#")[1]])
print(data[:10])
t = np.array(data, dtype=str)

trip = triples.TriplesFactory.from_labeled_triples(t)
training, testing = trip.split([0.95,0.05])

# path_to_kg = (r'C:\Users\dough\Documents\GitHub\KRoW-Project\data\Project.ttl')
# triples_factory = pykeen.triples.TriplesFactory.from_path(path=path_to_kg, create_inverse_triples=False)


embedding_model = pykeen.models.TransE(triples_factory = trip)

result = pipeline(
    random_seed = 0,
    model = embedding_model,
    training_kwargs = dict(num_epochs = 2),
    evaluation_kwargs = dict(),
    training=training,
    testing=testing,)

# Get the entity embeddings
entity_embeddings = result.model.get_all_entities()[0].detach().numpy()

# Cluster the embeddings using k-means
num_clusters = 5  # Set the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
cluster_labels = kmeans.fit_predict(entity_embeddings)

# Print the cluster labels
print("Cluster labels: ", cluster_labels)