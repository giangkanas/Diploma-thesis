import pandas as pd
from pecanpy import node2vec
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



""" ΔΗΜΙΟΥΡΓΙΑ EMBEDDINGS node2vec ΑΠΟ KARATECLUB"""  
from karateclub import Node2Vec

def emd_names_dic():
    # node_embeddings = pd.read_csv("./Embeddings/")
    lista = os.listdir("./Embeddings/KARATECLUB")
    emd_file_names_dic = {i : lista[i] for i in range(len(lista))}
    return emd_file_names_dic


def find_parameters(emdNum):
    dim = [16, 32, 64, 128, 256]
    num_walks = [10, 15, 20, 25]
    walk_length = [40, 60, 100, 120]
    P = [0.25, 0.5, 1., 2., 4.]
    Q = [0.25, 0.5, 1., 2., 4.]
    
    comb = []
    
    for i in range(emdNum):
        p = random.sample(P,k=1)[0]
        q = random.sample(Q,k=1)[0]
        d = random.sample(dim,k=1)[0]
        nw = random.sample(num_walks,k=1)[0]
        wl = random.sample(walk_length,k=1)[0]
        
        combination = [p,q,d,nw,wl]
        
        while combination in comb:
            p = random.sample(P,k=1)[0]
            q = random.sample(Q,k=1)[0]
            d = random.sample(dim,k=1)[0]
            nw = random.sample(num_walks,k=1)[0]
            wl = random.sample(walk_length,k=1)[0]
            
        comb.append(combination)
    return comb

def create_embeddings(graph,filename,params):
    
    p = params[0]
    q = params[1]
    d = params[2]
    nw = params[3]
    wl = params[4]
    
    graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None) # απαραιτητο για την βιβλιοθηκη karateclub
    model = Node2Vec(walk_number = nw , walk_length = wl, p = p , q = q, dimensions=d)
    model.fit(graph)
    embeddings = model.get_embedding()
    embeddings = pd.DataFrame(embeddings)
    filepath = './%s/Embeddings/emd_(p=%.2f_q=%.2f_dim=%d_numWalks=%d_walkLength=%d).csv'%(filename,p, q, d, nw, wl)
    embeddings.to_csv(filepath,index=False,header=None,sep = " ")
    
    
    # graph = nx.convert_node_labels_to_integers(graph, first_label=0, ordering='default', label_attribute=None) # απαραιτητο για την βιβλιοθηκη karateclub
    # model = Node2Vec()
    # model.fit(graph)
    # embeddings = model.get_embedding()
    # embeddings = pd.DataFrame(embeddings)
    # filepath = './'+filename+"/"+filename+"_emd.csv"
    # embeddings.to_csv(filepath,index=False,header=None,sep = " ")
    


if __name__ == "__main__":
    start_time = time.time()
    
    paths = ["./CollegeMsg/CollegeMsg.csv","./Facebook/facebook.txt","./Wiki-Vote/Wiki-Vote.txt"]
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
    directed = [True,False,True]
    

    # comb = find_parameters(10) Βρεθηκαν οι παρακάτω συνδιασμοί
    
    comb = [[0.25, 0.25, 64, 15, 60], [0.5, 0.25, 256, 10, 120], [0.25, 0.25, 16, 15, 60], [4.0, 4.0, 16, 15, 120],
            [0.5, 2.0, 256, 20, 40], [0.5, 2.0, 64, 20, 60], [0.5, 1.0, 16, 10, 120], [0.25, 1.0, 64, 20, 40], 
            [2.0, 4.0, 16, 25, 40], [4.0, 1.0, 256, 20, 60]]
    
    # print(comb)
    
    # for index,filename in enumerate(filenames):
    #     graph = read_network(paths[index])
    #     for j,i in enumerate(comb):
    #         create_embeddings(graph,filename,i)
    #         print(index,j)
    # graph = read_network(paths[2])
    
    
    graph = read_network(paths[1])
    for j,i in enumerate(comb):
        create_embeddings(graph,filenames[1],i)
        print(j)
    
    
    # create_embeddings(graph,filenames[2])

    



    
    
    print("--- %s seconds ---" % (time.time() - start_time))























