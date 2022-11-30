import pandas as pd
import time
import networkx as nx
from numpy.random import seed
import random
import os
# seed(1)


def read_network(filename):
    if filename=="./Facebook/facebook.txt":
        graph = nx.read_edgelist(filename,delimiter = " " , data=(("Type", str),),create_using=nx.Graph)
    elif filename =="./Wiki-Vote/Wiki-Vote.txt":graph = nx.read_edgelist(filename,delimiter = "\t" , data=(("Type", str),),create_using=nx.DiGraph)
    else:graph = nx.read_edgelist(filename,delimiter = " " , data=(("Type", str),),create_using=nx.DiGraph)
    
    return graph



""" ΔΗΜΙΟΥΡΓΙΑ EMBEDDINGS graph2vec ΑΠΟ KARATECLUB"""  
from karateclub.graph_embedding import Graph2Vec



import matplotlib.pyplot as plt

def create_embeddings(graph,filename,dim):
    graph = nx.convert_node_labels_to_integers(graph, first_label=1, ordering='default', label_attribute=None)
    node_subgraphs = []
    if graph.is_directed():
        nodes = list(graph.nodes()) # για καποιο λογο χωρις το List δεν το εμφανιζει στα variables
            
        
        for node in nodes:
            subgraph_edges1 = list(graph.in_edges(node)) # ακμες που πηγαινουν στον κόμβο
            # subgraph_edges2 = list(graph.out_edges(node)) # ακμες που φευγουν από τον κόμβο
            # subgraph_edges = subgraph_edges1+subgraph_edges2 #συνολικες ακμές κόμβου
            subgraph_edges = subgraph_edges1
            node_subgraph = nx.DiGraph(subgraph_edges)
            
            """ β)κανω rename τους κομβους καθε subset για να συμβαδιζει με το Indexing αυτο """
            subgraph_nodes = sorted(list(node_subgraph.nodes()))
            mapping = {i:j for j,i in enumerate(subgraph_nodes)}
            node_subgraph = nx.relabel_nodes(node_subgraph, mapping)
            
            node_subgraphs.append(node_subgraph)
            
            # nx.draw(node_subgraph,with_labels = True)
            # plt.show()
            # break
    else:
        nodes = list(graph.nodes()) # για καποιο λογο χωρις το List δεν το εμφανιζει στα variables
            
        for node in nodes:
            subgraph_edges = list(graph.edges(node)) # ακμες που πηγαινουν ή ερχονται από τον κόμβο
            
            node_subgraph = nx.Graph(subgraph_edges)
            
            """ β)κανω rename τους κομβους καθε subset για να συμβαδιζει με το Indexing αυτο """
            subgraph_nodes = sorted(list(node_subgraph.nodes()))
            mapping = {i:j for j,i in enumerate(subgraph_nodes)}
            node_subgraph = nx.relabel_nodes(node_subgraph, mapping)
            
            node_subgraphs.append(node_subgraph)
            
            # nx.draw(node_subgraph,with_labels = True)
            # plt.show()
            # break
        
    """ επειδη το graph2vec μου βγαζει το ακολουθο error (AssertionError: The node indexing is wrong.)
        α)εβγαλα το check_indexing και δεν δημιουργείται κάποιο πρόβλημα => ΔΕΝ ΧΡΕΙΑΣΤΗΚΕ ΓΙΑΤΙ ΛΕΙΤΟΥΡΓΗΣΕ ΤΟ Β
        β)κανω rename τους κομβους καθε subset για να συμβαδιζει με το Indexing αυτο"""
    
    model = Graph2Vec(attributed=False,dimensions=dim)
    
    model.fit(node_subgraphs)
    embeddings = model.get_embedding()
    embeddings = pd.DataFrame(embeddings)
    path = "./"+filename+"/Embeddings/"+filename+"_emd_"+str(dim)+".csv"
    embeddings.to_csv(path,index=False,header=None,sep = " ")
    
    
    
    
        


if __name__ == "__main__":
    start_time = time.time()
    
    paths = ["./CollegeMsg/CollegeMsg.csv","./Facebook/facebook.txt","./Wiki-Vote/Wiki-Vote.txt"]
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
    # directed = [True,False,True]
    
    dimensions = [16,32,64,128,256]
    
    for i,filename in enumerate(filenames):
        graph = read_network(paths[i])
        for dim in dimensions:
            create_embeddings(graph,filenames[i],dim)
            print(i,dim)

            
    
        
    

    



    
    
    print("--- %s seconds ---" % (time.time() - start_time))























