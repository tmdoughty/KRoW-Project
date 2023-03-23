# To use Scikit-learn (sklearn) and PyKEEN to perform KMeans clustering over a knowledge graph, you can follow these steps:

# Load the Knowledge Graph: First, you need to load the knowledge graph into a PyKEEN-compatible format. PyKEEN can load knowledge graphs 
# from various sources such as RDF triples, TSV files, and more. You can use the PyKEEN API to load your knowledge graph into a PyKEEN-compatible 
# format.

# Extract Features: After loading the knowledge graph, you need to extract features for each entity in the graph. PyKEEN provides several 
# methods for feature extraction such as TransE, RotatE, and more. You can use these methods to extract features for each entity.

# Train the KMeans Model: Once you have extracted the features, you can use the Scikit-learn API to train a KMeans model. You can pass the 
# extracted features to the KMeans model and specify the number of clusters you want to create.

# Evaluate the Model: After training the KMeans model, you can evaluate its performance by calculating metrics such as the Silhouette 
# Coefficient or Inertia.

# Here's an example code snippet that shows how to use PyKEEN and Scikit-learn to perform KMeans clustering over a knowledge graph:

import pykeen
from sklearn.cluster import KMeans
import sklearn 


# load the knowledge graph
knowledge_graph = from_path(r'C:\Users\dough\Documents\GitHub\KRoW-Project\data\Project_onto.owl')
# knowledge_graph = pykeen.triples.utils.load_triples(r'C:\Users\dough\Documents\GitHub\KRoW-Project\data\Project_onto.owl')
# pykeen.triples.utils.load_triples()
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