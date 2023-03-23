import pykeen
from sklearn.cluster import KMeans
import sklearn 

# load the knowledge graph
knowledge_graph = pykeen.load_triples(r'C:\Users\dough\Documents\GitHub\KRoW-Project\data\Project_onto.owl')

# extract features
model = pykeen.models.TransE()
model.fit(knowledge_graph)
features = model.get_all_entities()

# train the KMeans model
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(features)

# evaluate the model
silhouette_coefficient = sklearn.metrics.silhouette_score(features, kmeans.labels_)
inertia = kmeans.inertia_

print(f"Silhouette Coefficient: {silhouette_coefficient}")
print(f"Inertia: {inertia}")