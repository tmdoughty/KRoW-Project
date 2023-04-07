# you need to install pykeen beforehand, see https://pykeen.readthedocs.io/en/stable/installation.html 
import numpy as np
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen import triples
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
import matplotlib.pyplot as plt
import os

# parse the KG into a graph
g = Graph()
g.parse("data/KG.ttl")

############################# create triples from graph; should be np.array ########################################

# data = {'subject':[], 'predicate':[], 'object':[]}
data = []
i=0
for s,p,o in g.triples((None,None,None)):
    if '#' in s and '#' in p and '#' in o:
        data.append([s.split("#")[1], p.split("#")[1], o.split("#")[1]])
t = np.array(data, dtype=str)

############################################ ML as in tutorial ##############################################

# TRIPLES
trip = triples.TriplesFactory.from_labeled_triples(t)

# SPLIT DATA INTO TRAIN AND TEST
training, testing = trip.split([0.95,0.05])

print('Train set size: ', training.triples.shape)
print('Test set size: ', testing.triples.shape)


# CREATE PIPELINE
pipeline_result = pipeline(
    random_seed=0,
    model='ComplEx',                # CHANGE MODEL:
    #dimensions=150,                         # elise: ComplEx
    training=training,                      # Josip: TransE
    testing=testing,                        # taylor: RotatE
     training_kwargs=dict(                  # nikki: DistMult
        num_epochs=1,                                          # epochs: 50 and 100
        checkpoint_name='got_complex_checkpoint.pt',
        checkpoint_directory='checkpoint_dir/',
        checkpoint_frequency=20,
    ),
    optimizer='adam',                   
    optimizer_kwargs={'lr':1e-3},
    loss='pairwisehinge',               
    regularizer='LP', 
    regularizer_kwargs={'p':3, 'weight':1e-5}, 
    negative_sampler='basic',
    negative_sampler_kwargs=dict(
        filtered=True,
    )
)

# RESULTS --> CHANGE NAME AND NR. EPOCHS
pipeline_result.plot_losses()                           
print(pipeline_result.get_metric('mrr'))            # inverse_harmonic_mean_rank in results
print(pipeline_result.get_metric('hits@10'))
pipeline_result.save_to_directory("ComplEx_50")     # CHANGE TO YOUR MODEL AND NR. EPOCHS

plt.savefig('ComplEx_50_losses.png')                # CHANGE TO YOUR MODEL AND NR. EPOCHS
plt.show()  

############################################ MISSING LINK PREDICTION ##################################################
from pykeen import predict
from scipy.special import expit

# all drugs
drugcodes = []
symptoms = []
for triple in data:
    if 'resultedIn' in triple[1]:
        if triple[0] not in drugcodes:
            drugcodes.append(triple[0])
        if triple[2] not in symptoms:
            symptoms.append(triple[2])

links = []
for drug in drugcodes:
    for symptom in symptoms:
        links.append([drug, 'resultedIn', symptom])

X_unseen = np.array(links)

# from pykeen.predict import predict_triples

# got_unseen = triples.get_mapped_tripples(X_unseen,factory=got)
pack = predict.predict_triples(model=pipeline_result.model, triples=X_unseen, triples_factory=trip)

processed_results = pack.process().df
# print(processed_results)

probs = expit(processed_results['score'])
# print(probs)

processed_results['prob'] = probs
processed_results['triple'] = list(zip([' '.join(x) for x in X_unseen]))

# processed_results
res = pd.DataFrame(list(zip([' '.join(x) for x in X_unseen],  
                      np.squeeze(processed_results['score']),
                      np.squeeze(probs))), 
             columns=['statement', 'score', 'prob']).sort_values("score")

res.to_pickle("ComplEx_50.pkl")     # CHANGE TO YOUR MODEL AND NR. EPOCHS

## to read table
# df = pd.read_pickle("ComplEx_50.pkl")
# print(df)
