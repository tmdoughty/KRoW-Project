from rdflib import Graph

INSTANCES_AND_CLASSES_QUERY = """SELECT DISTINCT ?subject WHERE { ?subject rdf:type ?obj .} LIMIT 100"""

CLASSES_QUERY = """SELECT DISTINCT ?subject WHERE { ?subject rdf:type owl:Class .} LIMIT 100"""

ASK_PERSON_INSTANCE_EXISTS = """ASK { ?person rdf:type ex:Person .}"""

PEOPLE_INFO_STATIC = """SELECT ?person ?sex ?age ?bodyWeight ?height
WHERE {
?person ex:hasAge ?age .
?person ex:hasBodyWeight ?bodyWeight .
?person ex:hasHeight ?height .
?person ex:hasSex ?sex .
} ORDER BY ?person
LIMIT 100
"""

DRUG_INFO = """SELECT ?drug ?property ?test
WHERE {
?drug ex:hasComplexity ?complexity;
    ?property ?test.
}LIMIT 100
"""

DRUG_REACTIONS = """SELECT ?drug ?sideEffectCode ?HLTGName ?HLTName ?PTName ?SOCName
WHERE {
?drug ex:resultedIn ?sideEffectCode.
?sideEffectCode ex:hasHLTGName ?HLTGName ;
    ex:hasHLTName ?HLTName ;
    ex:hasPTName ?PTName ;
    ex:hasSOCName ?SOCName .
} ORDER BY ?drug
LIMIT 100
"""

CONSTRUCT_DRUG_SIDE_EFFECT_LIST = """
CONSTRUCT {?drug ex:hasSideEffect ?sideEffect}
WHERE {
    ?drug ex:resultedIn ?id.
    ?id ex:hasPTName ?sideEffect.
} LIMIT 100
"""

DESCRIBE_SPECIFIC_DRUG = """
DESCRIBE ex:N02AC01
"""


DESCRIBE_SPECIFIC_PERSON = """
DESCRIBE ex:Person7601
"""

# Loads data into graph object
def load_data():
    # parse the ontology into a graph
    g = Graph()
    # Project.ttl does not exist in git, renamed to Project
    g.parse("./data/Project")
    return g

# Runs query on given graph, prints results
def run_query(graph, desired_query):
    result = graph.query(desired_query)
    for row in result:
        print(row)

# Runs queries in given list
def main(desired_queries):
    project_graph = load_data()
    for query in desired_queries:
        print("\n\n")
        print("#"*8, "NEW QUERY", "#"*8)
        run_query(project_graph, query)

if __name__ == "__main__":
    ##### Queries #####
    # INSTANCES_AND_CLASSES_QUERY       ->      Give list of instances and classes
    # CLASSES_QUERY                     ->      Gives list of classes
    # ASK_PERSON_INSTANCE_EXISTS        ->      Asks if something of type person exists
    # PEOPLE_INFO_STATIC                ->      Gives info on person id, sex, age, weight, height
    # DRUG_INFO                         ->      Gives all available information for a specific drug (limited to 150 results)
    # DRUG_REACTIONS                    ->      Gives information of drug side effects (limited to 500 results)
    # CONSTRUCT_DRUG_SIDE_EFFECT_LIST   ->      Constructs graph of side effects & drugs
    # DESCRIBE_SPECIFIC_DRUG            ->      Describes a given drug 
    # DESCRIBE_SPECIFIC_PERSON          ->      Describes a given person

    ### Input ###
    # main([list_of_queries])

    ### Output ###
    # Printed out list. Could be saved somewhere if we want.

    ### Examples here. Uncomment / comment out whatever you want. ###

    # main([INSTANCES_AND_CLASSES_QUERY, CLASSES_QUERY, ASK_PERSON_INSTANCE_EXISTS])
    # main([ASK_PERSON_INSTANCE_EXISTS])
    # main([PEOPLE_INFO_STATIC])
    # main([DRUG_INFO])
    # main([DRUG_REACTIONS])
    # main([INSTANCES_AND_CLASSES_QUERY, CLASSES_QUERY, PEOPLE_INFO_STATIC, DRUG_INFO, DRUG_REACTIONS])
    main([CONSTRUCT_DRUG_SIDE_EFFECT_LIST, DESCRIBE_SPECIFIC_DRUG, DESCRIBE_SPECIFIC_PERSON])
