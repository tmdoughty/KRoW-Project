# you need to install pykeen beforehand, see https://pykeen.readthedocs.io/en/stable/installation.html 
import numpy as np
import pandas as pd
import seaborn as sb

from pykeen.datasets import Nations
from pykeen import triples

import rdflib
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from rdflib.namespace import DC, FOAF
import matplotlib.pyplot as plt

# parse the ontology into a graph
g = Graph()
g.parse("./data/Project.ttl")

############################# create triples from graph; should be np.array ########################################

# data = {'subject':[], 'predicate':[], 'object':[]}
data = []
i=0
for s,p,o in g.triples((None,None,None)):
    if '#' in s:
        s = s.split("#")[1]
    if '#' in p:
        p = p.split("#")[1]
    if '#' in o:
        o = o.split("#")[1]
    data.append([s, p, o])
t = np.array(data, dtype=str)


############################################### ML + CLUSTERING  ###################################################

# ml and clustering from https://docs.ampligraph.org/en/1.1.0/tutorials/ClusteringAndClassificationWithEmbeddings.html
# not finished, issues with import statements
# packages necessary: tensorflow, ampligraph, ..
import ampligraph
from ampligraph.evaluation import train_test_split_no_unseen

X_train, X_valid = train_test_split_no_unseen(t, test_size=5)

from ampligraph.latent_features import ComplEx

# prediction model
model = ComplEx(batches_count=10,
                epochs=300,
                k=100,
                eta=20,
                optimizer='adam', 
                optimizer_params={'lr':1e-4},
                loss='multiclass_nll',
                regularizer='LP', 
                regularizer_params={'p':3, 'lambda':1e-5}, 
                seed=0, 
                verbose=True)

# train model
model.fit(X_train)

# filter out false negatives generted by the corruption strategy
filter_triples = np.concatenate((X_train, X_valid))

# evaluate performance
from ampligraph.evaluation import evaluate_performance

ranks = evaluate_performance(X_valid,
                             model=model, 
                             filter_triples=filter_triples,
                             use_default_protocol=True,
                             verbose=True)

from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score

mr = mr_score(ranks)
mrr = mrr_score(ranks)

print("MRR: %.2f" % (mrr))
print("MR: %.2f" % (mr))

#hits at n score: the probability that the model will put the correct value
# in the thop n possibilities
hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % (hits_10))
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % (hits_1))


############################################ ML as in tutorial ##############################################

# transform data into triples factory
trip = triples.TriplesFactory.from_labeled_triples(t)

#split data into test and train set
training, testing = trip.split([0.95,0.05])

print('Train set size: ', training.triples.shape)
print('Test set size: ', testing.triples.shape)

from pykeen.pipeline import pipeline

# here we don't import the model, but let PyKEEN do the importing.
pipeline_result = pipeline(
    random_seed=0,
    model='HolE',
    training=training,
    testing=testing,
    #  training_kwargs=dict(
    #     num_epochs=200,
    #     checkpoint_name='got_complex_checkpoint.pt',
    #     checkpoint_directory='checkpoint_dir/',
    #     checkpoint_frequency=20,
    # ),
    # dimensions=150,
    # optimizer='adam',
    # optimizer_kwargs={'lr':1e-3},
    # loss='pairwisehinge', 
    # regularizer='LP', 
    # regularizer_kwargs={'p':3, 'weight':1e-5}, 
    # negative_sampler='basic',
    # negative_sampler_kwargs=dict(
    #     filtered=True,
    # )
)
pipeline_result.plot_losses()
plt.show()
print(pipeline_result.get_metric('mrr'))
print(pipeline_result.get_metric('hits@10'))
pipeline_result.save_to_directory("./data/results")
