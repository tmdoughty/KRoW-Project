import pandas as pd
import rdflib
from rdflib import Graph, ConjunctiveGraph, Literal, BNode, Namespace, RDF, URIRef, Literal, OWL, RDFS
from rdflib.namespace import DC, FOAF
from owlrl import DeductiveClosure, RDFS_Semantics, OWLRL_Semantics 

import networkx as nx
from owlready2 import get_ontology, Thing, DataProperty, ObjectProperty, sync_reasoner

data =  pd.read_csv('data/opioids_data.csv', sep=',',engine='python')

additional_data = pd.read_csv('data/additional_data.csv', sep=',',engine='python')

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
class ATCName(Codes):
    namespace = onto


# drug properties
class MuOpiodReceptor(Thing):
    namespace = onto
class SodiumDependentNoradrenalineTransporter(Thing):
    namespace = onto
class SodiumDependentSerotoninTransporter(Thing):
    namespace = onto
class DeltaOpiodReceptor(Thing):
    namespace = onto
class KappaOpiodReceptor(Thing):
    namespace = onto
class MolecularWeight(Thing):
    namespace = onto
class XLogP3(Thing):
    namespace = onto
class HydrogenBondDonorCount(Thing):
    namespace = onto
class HydrogenBondAcceptorCount(Thing):
    namespace = onto
class RotatableBondCount(Thing):
    namespace = onto
class MonoisotopicMass(Thing):
    namespace = onto
class HeavyAtomCount(Thing):
    namespace = onto
class Complexity(Thing):
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
class drugName(ObjectProperty):
    domain = [ATCode]
    range = [ATCName]
    namespace = onto


# drug properties

class hasMuOpiodReceptor:
    domain = [ATCode]
    range = [MuOpiodReceptor]
    namespace = onto

class hasSodiumDependentNoradrenalineTransporter:
    domain = [ATCode]
    range = [SodiumDependentNoradrenalineTransporter]
    namespace = onto

class hasSodiumDependentSerotoninTransporter:
    domain = ATCode
    range = [SodiumDependentSerotoninTransporter]
    namespace = onto

class hasDeltaOpiodReceptor:
    domain = ATCode
    range = [DeltaOpiodReceptor]
    namespace = onto

class hasKappaOpiodReceptor:
    domain = ATCode
    range = [KappaOpiodReceptor]
    namespace = onto

class hasMolecularWeight:
    domain = [ATCode]
    range = [MolecularWeight]
    namespace = onto

class hasXLogP3:
    domain = [ATCode]
    range = [XLogP3]
    namespace = onto

class hasHydrogenBondDonorCount:
    domain = [ATCode]
    range = [HydrogenBondDonorCount]
    namespace = onto

class hasHydrogenBondAcceptorCount:
    domain = [ATCode]
    range = [HydrogenBondAcceptorCount]
    namespace = onto

class hasRotatableBondCount:
    domain = [ATCode]
    range = [RotatableBondCount]
    namespace = onto

class hasMonoisotopicMass:
    domain = [ATCode]
    range = [MonoisotopicMass]
    namespace = onto

class hasHeavyAtomCount:
    domain = [ATCode]
    range = [HeavyAtomCount]
    namespace = onto

class hasComplexity:
    domain = [ATCode]
    range = [Complexity]
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
    g.add((atc, EX.drugName, Literal(opioid['ATCText'])))


for index, opioid in additional_data.iterrows():
    drug = URIRef(EX+opioid["ATCode"])
    g.add((drug, EX.hasMuOpiodReceptor, Literal(opioid["Mu-type opioid receptor"])))
    g.add((drug, EX.hasSodiumDependentNoradrenalineTransporter, Literal(opioid["Sodium-dependent noradrenaline transporter"])))
    g.add((drug, EX.hasSodiumDependentNoradrenalineTransporter, Literal(opioid["Sodium-dependent noradrenaline transporter"])))
    g.add((drug, EX.hasSodiumDependentSerotoninTransporter, Literal(opioid["Sodium-dependent serotonin transporter"])))
    g.add((drug, EX.hasDeltaOpioidReceptor, Literal(opioid["Delta-type opioid receptor"])))
    g.add((drug, EX.hasKappaOpioidReceptor, Literal(opioid["Kappa-type opioid receptor"])))
    g.add((drug, EX.hasMolecularWeight, Literal(opioid["molecular weight"])))
    g.add((drug, EX.hasXLogP3, Literal(opioid["XLogP3"])))
    g.add((drug, EX.hasHydrogenBondDonorCount, Literal(opioid["Hydrogen Bond Donor Count"])))
    g.add((drug, EX.hasHydrogenBondAcceptorCount, Literal(opioid["Hydrogen Bond Acceptor Count"])))
    g.add((drug, EX.hasRotatableBondCount, Literal(opioid["Rotatable Bond Count"])))
    g.add((drug, EX.hasMonoisotopicMass, Literal(opioid["Monoisotopic Mass"])))
    g.add((drug, EX.hasMHeavyAtomCount, Literal(opioid["Heavy Atom Count"])))
    g.add((drug, EX.hasComplexity, Literal(opioid["Complexity"])))
    
    

#### save graph ####
g.serialize(format='turtle', destination="./data/KG.ttl")




