# KRoTW project - Group 1b

## How to set up work environment

Using pip, or another package manager install all the requirements within the requirements.txt file

For pip:
```
pip install -r requirements.txt
```


## How to start experiments

### Step 0: Prerequisite 
In order to recreate the experiments made within the paper, one of the first things to be done is running the krw_ontology.py file. (note: in case of 'file-not-found error' make sure both the opioids.csv and additional_data.csv are both correctly spelled)

### Step 1: SPARQL
The SPARQL queries were created and used within the krw_sparql_exploration.py file. To run krw_sparql_exploration.py one must run the main function within it, which takes a list of SPARQL queries defined at the top of the file. The only requirement within the file that it is pointing to the correct .ttl file (either 'KG.ttl' or 'Project' within the data folder). 
