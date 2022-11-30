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
    if filename=="facebook.txt":
        graph = nx.read_edgelist(filename,delimiter = " " , data=(("Type", str),),create_using=nx.Graph)
    elif filename =="Wiki-Vote.txt":graph = nx.read_edgelist(filename,delimiter = "\t" , data=(("Type", str),),create_using=nx.DiGraph)
    else:graph = nx.read_edgelist(filename,delimiter = " " , data=(("Type", str),),create_using=nx.DiGraph)
    
    return graph


def find_spreads(filename,g,fraction_infected,beta,number_of_spreads):
    total_spreads = []
    total_root_nodes = []
    for i in range(number_of_spreads):
        root_nodes = rootRandomly(g,fraction_infected)
        while(root_nodes in total_root_nodes):
            root_nodes = sorted(rootRandomly(g,fraction_infected))
        
        total_root_nodes.append(root_nodes)
        spread , model, iterations = SIModel(g,beta,root_nodes)
        percentage = iterations[-1]["node_count"][1]/(iterations[-1]["node_count"][0]+iterations[-1]["node_count"][1])*100
        percentage = "{:.2f}".format(percentage)
        total_spreads.append((len(spread),percentage,spread,root_nodes))
        # total_spreads.append((len(spread),root_nodes,spread))
        
        df = pd.DataFrame(total_spreads)
        df.to_csv(filename+"_spreads.csv",index=False,header=None,sep = " ")
        print(i,percentage)



def SIModel(G,beta,root_nodes):

    
    # Model selection
    model = ep.SIModel(G)    
    # Model Configuration
    
    config = mc.Configuration()
    config.add_model_parameter('beta', beta)
    config.add_model_initial_configuration("Infected", root_nodes)
    
    model.set_initial_status(config)
    
   
    # Simulation interaction graph based execution
    iterations = model.iteration_bunch(3)
    
    spread = [i for i in root_nodes]
    for iteration in iterations:
        if iteration["iteration"]!=0:
            newActive = [i for i in list(iteration["status"].keys())]
            spread += newActive
    
    
    return spread , model , iterations



if __name__ == '__main__':
    start_time = time.time()
    directed = [True, False ,True]
    #filenames = ["Wiki-Vote"]
    filenames = ["CollegeMsg","facebook","Wiki-Vote"]
    paths = ["CollegeMsg.csv","facebook.txt","Wiki-Vote.txt"]

    
    for i,filename in enumerate(filenames):
        
        g  = read_network(paths[i])
        fraction_infected = 0.05 # percentage of initial infected nodes
        beta = 0.1 # probability of influence
                
        find_spreads(filename,g,fraction_infected,beta,500)
 
        
        
        
    print("--- %s seconds ---" % (time.time() - start_time))
    
    
    
    
    
    
    
    
    
    
    
    