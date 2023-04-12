# KRoTW project - Group 1b

## How to set up work environment

Using pip, or another package manager install all the requirements within the requirements.txt file

For pip:
```
pip install -r requirements.txt
```


## How to start experiments

### Prerequisite 
In order to recreate the experiments made within the paper, one of the first things to be done is running the Ontology.py file. (note: in case of 'file-not-found error' make sure both the opioids.csv and additional_data.csv are both correctly spelled), The Ontology.py file is the first thing that has to be run in order for all other files to work.

### SPARQL
The SPARQL queries were created and used within the krw_sparql_exploration.py file. To run krw_sparql_exploration.py one must run the main function within it, which takes a list of SPARQL queries defined at the top of the file. The only requirement within the file that it is pointing to the correct .ttl file (either 'KG.ttl' or 'Project' within the data folder). 

### Clustering
- Louvain: run LouvainClustering.py
- 
- 

### Machine Learning
Run the ML.py file. It only runs one PyKEEN model and one epoch size. To get the other, it needs to be changed by hand in the pipeline function and in the functions that save results.

### Missing Link Prediction
MLP is performed in the ML.py file. To obtain the results, run the MLP.ipynb file which prints the probabilities for all links for one model and one epoch size. Change the name of the pickle file to obtain the results for other models/epoch sizes.
