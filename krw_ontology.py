import pandas as pd
import rdflib
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from rdflib.namespace import DC, FOAF
from owlrl import DeductiveClosure, RDFS_Semantics, OWLRL_Semantics 

import networkx as nx
from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty, sync_reasoner

data =  pd.read_csv('./data/opioids.csv', sep=',',engine='python')

onto = get_ontology("http://test.org/onto.owl")

#### create classes ####

# main
class Person(Thing):
    namespace = onto
class Codes(Thing):
    namespace = onto
class LLTCode(Codes):
    namespace = onto

# person data
class BodyWeight(Thing):
    namespace = onto
class Height(Thing):
    namespace = onto
class Sex(Thing):
    namespace = onto
class Age(Thing):
    namespace = onto

# drug
class ATCode(Codes):
    namespace = onto

# symptoms
class PTCode(Codes):
    namespace = onto
class HLTCode(Codes):
    namespace = onto
class HLTGCode(Codes):
    namespace = onto
class SOCCode(Codes):
    namespace = onto

#### create properties ####

# person
class hasBodyWeight(DataProperty):
    domain = [Person]
    range = [BodyWeight]
    namespace = onto
class hasHeight(DataProperty):
    domain = [Person]
    range = [Height]
    namespace = onto
class hasSex(DataProperty):
    domain = [Person]
    range = [Sex]
    namespace = onto
class hasAge(DataProperty):
    domain = [Person]
    range = [Age]
    namespace = onto

# symptom
class hasPTName(ObjectProperty):
    domain = [PTCode]
    range = [str]
    namespace = onto
class hasHLTName(ObjectProperty):
    domain = [HLTCode]
    range = [str]
    namespace = onto
class hasHLTGName(ObjectProperty):
    domain = [HLTGCode]
    range = [str]
    namespace = onto
class hasSOCName(ObjectProperty):
    domain = [SOCCode]
    range = [str]
    namespace = onto


class hasPTCode(ObjectProperty):
    domain = [LLTCode]
    range = [PTCode]
    namespace = onto
class hasHLTCode(ObjectProperty):
    domain = [LLTCode]
    range = [HLTCode]
    Codespace = onto
class hasHLTGCode(ObjectProperty):
    domain = [LLTCode]
    range = [HLTGCode]
    namespace = onto
class hasSOCCode(ObjectProperty):
    domain = [LLTCode]
    range = [SOCCode]
    namespace = onto
    
class drugDosage(DataProperty):
    domain = [ATCode]
    range = [str]
    namespace = onto
class resultedIn(ObjectProperty):
    domain = [ATCode]
    range = [LLTCode]
    namespace = onto
class occuredIn(ObjectProperty):
    domain = [LLTCode]
    range = [Person]
    namespace = onto


#### save ontology ####
onto.save(file = "./data/Project_onto.owl", format = "rdfxml")

#### create instances ####

EX = rdflib.Namespace("http://test.org/myonto.owl#")
g = Graph()
g.parse("./data/Project_onto.owl")
g.bind("ex", EX)

for index, opioid in data.iterrows():
    per = URIRef(EX+"Person"+str(index))
    #g.add((uri, RDF.type, EX.Person))
    g.add((per, EX.hasBodyWeight, Literal(opioid["BodyWeight"])))
    g.add((per, EX.hasHeight, Literal(opioid["Height"])))
    g.add((per, EX.hasSex, Literal(opioid["sex"])))
    g.add((per, EX.hasAge, Literal(opioid["age_year"])))

    llt = URIRef(EX+str(opioid["LLTCode"]))
    g.add((llt, EX.occuredIn, per))
    g.add((llt, EX.hasPTName, Literal(opioid["PTName"])))
    g.add((llt, EX.hasHLTName, Literal(opioid["HLTName"])))
    g.add((llt, EX.hasHLTGName, Literal(opioid["HLTGName"])))
    g.add((llt, EX.hasSOCName, Literal(opioid["SOCName"])))

    atc = URIRef(EX+opioid["ATCode"])
    g.add((atc, EX.resultedIn, llt))
    g.add((atc, EX.drugDosage, Literal(opioid["GenericDrugName"])))
    

#### reasoner ####
DeductiveClosure(OWLRL_Semantics ).expand(g)

#### save graph ####
g.serialize(format='turtle', destination="./data/Project")




