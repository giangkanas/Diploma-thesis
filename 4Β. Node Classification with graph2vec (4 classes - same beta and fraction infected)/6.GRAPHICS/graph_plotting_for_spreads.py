import networkx as nx

import ndlib.models.ModelConfig as mc
import ndlib.models.epidemics as ep
import pandas as pd
import time
import random

def myFunc2(e):
    return -e[3]

def myFunc3(e):
    # print(e[1])
    return -e[1]

def rootRandomly(G,fraction_infected):
    node_number = int(G.number_of_nodes() * fraction_infected + 0.5)
    root_nodes = random.sample(G.nodes(),node_number)
    return root_nodes
    
# =============================================================================================
# =========================== READ NETWORK ================================================



def read_network(filename):
    if filename=="facebook":
        graph = nx.read_edgelist('./facebook/'+filename+'.txt',delimiter = " " , data=(("Type", str),),create_using=nx.Graph)
    elif filename =="Wiki-Vote":graph = nx.read_edgelist('./Wiki-Vote/'+filename+'.txt',delimiter = "\t" , data=(("Type", str),),create_using=nx.DiGraph)
    else:graph = nx.read_edgelist('./CollegeMsg/'+filename+'.csv',delimiter = " " , data=(("Type", str),),create_using=nx.DiGraph)
    
    return graph

def graph_plotting_for_spread_length():
    
    filenames = ["CollegeMsg","facebook","Wiki-Vote"]
    paths = ["./CollegeMsg/CollegeMsg_spreads.csv","./facebook/facebook_spreads.csv","./Wiki-Vote/Wiki-Vote_spreads.csv"]
   
    for i,filename in enumerate(filenames):
        
        spreads = pd.read_csv(paths[i],delimiter=" ",header=None)
        length = spreads[0]
                
        fig = plt.figure()
        plt.plot(length)
        
        plt.title(filename+'_spreads')
        plt.ylabel('spread size')
        plt.xlabel('SI simulation ID')
        
        plt.show()

        fig.savefig('./'+filename+'/'+filename+'_spread_length.png' , dpi=500)

def graph_plotting_for_node_prob_of_being_infected():
    filenames = ["CollegeMsg","facebook","Wiki-Vote"]
    paths = ["./CollegeMsg/CollegeMsg_prob_of_being_infected.csv","./facebook/facebook_prob_of_being_infected.csv","./Wiki-Vote/Wiki-Vote_prob_of_being_infected.csv"]
    for i,filename in enumerate(filenames):
        
        probs = pd.read_csv(paths[i],delimiter=" ",header=None)
        length = probs[0]
                
        fig = plt.figure()
        plt.plot(length)
        
        plt.title(filename+' node influence')
        plt.ylabel('probability of being infected')
        plt.xlabel('nodes')
        
        plt.show()

        fig.savefig('./'+filename+'/'+filename+'_node_influence.png' , dpi=500)
        
def graph_plotting_node_labels():
    filenames = ["CollegeMsg","facebook","Wiki-Vote"]
    paths = ["./CollegeMsg/CollegeMsg_dataset.csv","./facebook/facebook_dataset.csv","./Wiki-Vote/Wiki-Vote_dataset.csv"]
   
    for i,filename in enumerate(filenames):
        
        dataset = pd.read_csv(paths[i],delimiter=" ",header=None) 
        labels = pd.DataFrame(dataset.iloc[: , -1])
        labels.columns = range(labels.columns.size)
        labels.columns = range(labels.shape[1])
        labels = labels.rename({0:"Nodes per label"} , axis = 1)
         
        count = labels["Nodes per label"].groupby(labels["Nodes per label"]).count()
        
        
        count.plot.pie(autopct='%1.0f%%',
            title = filename +" nodes per label").get_figure().savefig(
                './'+filename+'/'+filename+'_node_labels.png',dpi = 500)
        
        plt.show()        
        
        
        # fig.savefig('./'+filename+'/'+filename+'_node_labels.png' , dpi=500)

def spread_dataset():
    filenames = ["CollegeMsg","facebook","Wiki-Vote"]
    paths = ["./CollegeMsg/CollegeMsg_spreads.csv","./facebook/facebook_spreads.csv","./Wiki-Vote/Wiki-Vote_spreads.csv"]
   
    for i,filename in enumerate(filenames):
        
        spreads = pd.read_csv(paths[i],delimiter=" ",header=None) 
        spreads = spreads.drop([3],axis=1)
        spreads = spreads.drop([1],axis=1)
        spreads = spreads.rename(columns = {0 : 1})
        print(spreads.head(10))
        print()
        
def final_dataset():
    filenames = ["CollegeMsg","facebook","Wiki-Vote"]
    paths = ["./CollegeMsg/CollegeMsg_dataset.csv","./facebook/facebook_dataset.csv","./Wiki-Vote/Wiki-Vote_dataset.csv"]
   
    for i,filename in enumerate(filenames):
        
        dataset = pd.read_csv(paths[i],delimiter=" ",header=None) 
        print()
        print(dataset.head(10))
        print()
    

import matplotlib.pyplot as plt
if __name__ == '__main__':
    start_time = time.time()
    
    # graph_plotting_for_spread_length()
    
    
    # graph_plotting_for_node_prob_of_being_infected()
    
    graph_plotting_node_labels()
        
    # spread_dataset()
    final_dataset()    
        
    print("--- %s seconds ---" % (time.time() - start_time))
        
    
    
    
    

    





    
    


    
    
    
    
    
    
    