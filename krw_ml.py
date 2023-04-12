import numpy as np
import pandas as pd
from pykeen.pipeline import pipeline
from pykeen import triples
from rdflib import Graph
import matplotlib.pyplot as plt
from pykeen import predict
from scipy.special import expit

#### create data for ML ####

# parse the KG into a graph
g = Graph()
g.parse("data/KG.ttl")

# get all triples
data = []
for s,p,o in g.triples((None,None,None)):
    if '#' in s:
        s = s.split("#")[1]
    if '#' in p:
        p = p.split("#")[1]
    if '#' in o:
        o = o.split("#")[1]
    data.append([s, p, o])
t = np.array(data, dtype=str)

##### ML as in tutorial ####

# put triples in correct format
trip = triples.TriplesFactory.from_labeled_triples(t)

# split data into train and test set
training, testing = trip.split([0.95,0.05])

print('Train set size: ', training.triples.shape)
print('Test set size: ', testing.triples.shape)


# create pipeline
pipeline_result = pipeline(
    random_seed=0,
    model='DistMult',             # change model name tot run different model   
    dimensions=150,                         
    training=training,                      
    testing=testing,                        
     training_kwargs=dict(                  
        num_epochs=100,                         # change epochs, either 50 or 100                 
        checkpoint_name='checkpoint.pt',
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

# results
pipeline_result.plot_losses()                           
print(pipeline_result.get_metric('mrr'))            # inverse_harmonic_mean_rank in results
print(pipeline_result.get_metric('hits@10'))
pipeline_result.save_to_directory("DistMult_100")     

plt.savefig('DistMult_100_losses.png')                
plt.show()  

#### missing link prediction ####

# all drugs and symptoms
drugcodes = []
symptoms = []
for triple in data:
    if 'resultedIn' in triple[1]:
        if triple[0] not in drugcodes:
            drugcodes.append(triple[0])
        if triple[2] not in symptoms:
            symptoms.append(triple[2])

# all possible drug-symptom combinations
links = []
for drug in drugcodes:
    for symptom in symptoms:
        links.append([drug, 'resultedIn', symptom])

X_unseen = np.array(links)

# prediction + convert scores to probabilities
pack = predict.predict_triples(model=pipeline_result.model, triples=X_unseen, triples_factory=trip)
processed_results = pack.process().df
probs = expit(processed_results['score'])
processed_results['prob'] = probs
processed_results['triple'] = list(zip([' '.join(x) for x in X_unseen]))

# processed_results
res = pd.DataFrame(list(zip([' '.join(x) for x in X_unseen],  
                      np.squeeze(processed_results['score']),
                      np.squeeze(probs))), 
             columns=['statement', 'score', 'prob']).sort_values("score")

# save results in pickle file
res.to_pickle("DistMult_100.pkl")     

