# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:23:00 2022

@author: Giannis
"""

import pandas as pd
from jenkspy import JenksNaturalBreaks
import matplotlib.pyplot as plt
import networkx as nx
""" ==========================  jenks  ======================================="""

########################################################################
""" labels 2 """

def read_network(filename):
    if filename=="./Facebook/facebook.txt":
        graph = nx.read_edgelist(filename,delimiter = " " , data=(("Type", str),),create_using=nx.Graph)
    elif filename =="./Wiki-Vote/Wiki-Vote.txt":graph = nx.read_edgelist(filename,delimiter = "\t" , data=(("Type", str),),create_using=nx.DiGraph)
    else:graph = nx.read_edgelist(filename,delimiter = " " , data=(("Type", str),),create_using=nx.DiGraph)
    
    return graph

def index_node_dic(graph):
    
    nodes = graph.nodes()
    dic = {int(node):index for index, node in enumerate(nodes)}
    reverse_dic = {index:int(node) for index, node in enumerate(nodes)}
    return dic , reverse_dic
    

def ProbabilityOfInfectionForEachNode(filename,graph):
    """ ΒΡΙΣΚΩ ΓΙΑ ΚΑΘΕ ΚΟΜΒΟ ΤΗΝ ΠΙΘΑΝΟΤΗΤΑ ΝΑ ΓΙΝΕΙ INFECTED 
    Pn(infected) = (συνολο παραδειγματων SI που n εγινε infected) / (συνολο παραδειγματων S I που εξεταστηκαν)"""
    
    spreadDataset = pd.read_csv(filename+"_spreads.csv",delimiter=" ",header=None)
    spreadDataset[2] = spreadDataset[2].fillna("[]").apply(lambda x: eval(x))
    spreads = spreadDataset[2].to_list()
    
    dic , reverse_dic = index_node_dic(graph)
    
    
    
    
    
    total = [0 for i in range(graph.number_of_nodes())]
    for spread in spreads:
        for node in spread:
            total[dic[int(node)]]+=1
    
    
    probabilityInfectedDic = {reverse_dic[i] : total[i]/len(spreads) for i in range(len(total))}
    
    # probabilityInfectedDic = pd.DataFrame([total[i]/len(spreads) for i in range(nodes)])
    probabilityInfected = pd.DataFrame(probabilityInfectedDic.values())
    
    return probabilityInfected



def SetLabels(filename,graph,classes):
    """ ΓΙΑ ΚΑΘΕ ΚΟΜΒΟ ΒΡΙΣΚΩ ΤΗΝ ΠΙΘΑΝΟΤΗΤΑ ΝΑ ΓΙΝΕΙ infected. ΣΤΗΝ ΣΥΝΕΧΕΙΑ
    ΜΕ ΤΗΝ ΤΕΧΝΙΚΗ jenks natural breaks ΧΩΡΙΣΑ ΤΙΣ ΠΙΘΑΝΟΤΗΤΕΣ ΑΥΤΕΣ ΣΕ ν ΚΑΤΗΓΟΡΙΕΣ"""
    active = ProbabilityOfInfectionForEachNode(filename,graph)
    array = active[0].to_list()
    jnb = JenksNaturalBreaks(nb_class = classes)
    
    jnb.fit(array)
    predict = jnb.predict(array)
    active[1] = predict
    
    labels = pd.DataFrame(active[1])
    
    count = labels[1].groupby(labels[1]).count()
    
    labels.to_csv(filename+"_labels.csv",index=False, header=None)
    
    return labels , count


if "__main__":
    filenames = ["./CollegeMsg/CollegeMsg","./Facebook/facebook","./Wiki-Vote/Wiki-Vote"]
    paths = ["./CollegeMsg/CollegeMsg.csv","./Facebook/facebook.txt","./Wiki-Vote/Wiki-Vote.txt"]
    

    # dic = index_node_dic(graph)
    
    for j,filename in enumerate(filenames):
        graph = read_network(paths[j])
        labels , count = SetLabels(filename,graph,classes=3)
        # break
    # graph = read_network(filenames[0]+".csv", directed[0])
    # optimal_number_of_clusters(filenames[0],graph)
    # labels , count = SetLabels(filenames[2],graph,classes=3)
    
    
    
    

    





