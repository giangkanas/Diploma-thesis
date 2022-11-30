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



""" ΔΗΜΙΟΥΡΓΙΑ EMBEDDINGS DeepWalk ΑΠΟ KARATECLUB"""  
from karateclub.node_embedding.neighbourhood import Walklets

def emd_names_dic():
    # node_embeddings = pd.read_csv("./Embeddings/")
    lista = os.listdir("./Embeddings/KARATECLUB")
    emd_file_names_dic = {i : lista[i] for i in range(len(lista))}
    return emd_file_names_dic


def find_parameters(emdNum):
    dim = [4, 8, 16, 32, 64]
    num_walks = [10, 15, 20, 25]
    walk_length = [40, 60, 100, 120]
    
    
    comb = []
    
    for i in range(emdNum):
        
        d = random.sample(dim,k=1)[0]
        nw = random.sample(num_walks,k=1)[0]
        wl = random.sample(walk_length,k=1)[0]
        
        combination = [d,nw,wl]
        
        while combination in comb:
            
            d = random.sample(dim,k=1)[0]
            nw = random.sample(num_walks,k=1)[0]
            wl = random.sample(walk_length,k=1)[0]
            
        comb.append(combination)
    return comb

def create_embeddings(graph,filename,params):
    
    """ ΠΡΟΣΟΧΗ. ΣΕ ΑΥΤΟΝ ΤΟΝ ΑΛΓΟΡΙΘΜΟ ΠΟΛΛΑΠΛΑΣΙΑΖΕΤΑΙ ΤΟ window_size ΜΕ ΤΟ dimensions. 
    ΟΠΟΤΕ ΑΦΟΥ window_size=4 ΒΑΖΩ dim = dim/4 ΓΙΑ ΝΑ ΠΡΟΚΥΨΟΥΝ ΔΙΑΝΥΣΜΑΤΑ ΔΙΑΣΤΑΣΕΩΝ ΠΟΥ ΘΕΛΩ"""
    
    d = params[0]
    nw = params[1]
    wl = params[2]
    
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None) # απαραιτητο για την βιβλιοθηκη karateclub
    model = Walklets(walk_number = nw , walk_length = wl, dimensions=d)
    model.fit(graph)
    embeddings = model.get_embedding()
    embeddings = pd.DataFrame(embeddings)
    filepath = './%s/Embeddings/emd_(dim=%d_numWalks=%d_walkLength=%d).csv'%(filename, d*4, nw, wl)
    embeddings.to_csv(filepath,index=False,header=None,sep = " ")
    
    
    


if __name__ == "__main__":
    start_time = time.time()
    
    paths = ["./CollegeMsg/CollegeMsg.csv","./Facebook/facebook.txt","./Wiki-Vote/Wiki-Vote.txt"]
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
    directed = [True,False,True]
    

    # comb = find_parameters(10) #Βρεθηκαν οι παρακάτω συνδιασμοί
    
    comb = [[4, 25, 60], [4, 25, 100], [8, 10, 100], [8, 10, 60],
            [16, 25, 100], [16, 20, 40], [32, 20, 120], [32, 20, 100],
            [64, 15, 40], [64, 25, 120]]
    
    # print(comb)
    
    for index,filename in enumerate(filenames):
        graph = read_network(paths[index])
        for j,i in enumerate(comb):
            create_embeddings(graph,filename,i)
            print(index,j)
    
    
    
    # graph = read_network(paths[0])
    # for j,i in enumerate(comb):
    #     create_embeddings(graph,filenames[0],i)
    #     print(j)
    
    
    # create_embeddings(graph,filenames[2])

    



    
    
    print("--- %s seconds ---" % (time.time() - start_time))























