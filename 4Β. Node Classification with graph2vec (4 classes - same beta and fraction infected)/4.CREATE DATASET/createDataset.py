""" ΣΕ ΑΥΤΟ ΤΟ ΑΡΧΕΙΟ ΔΟΚΙΜΑΖΩ ΤΑ LABELS13,14,15,16 LABELS23,24,25,26 LABELS3, LABELS4 ME ENA MONTELO ΚΑΤΗΓΟΡΙΟΠΟΙΗΣΗΣ
ΓΙΑ ΚΑΘΕ EMBEDDING """
""" ΤΑ ΚΑΛΥΤΕΡΑ ΑΠΟΤΕΛΕΣΜΑΤΑ ΠΡΟΕΚΥΨΑΝ ΑΠΟ ΤΟ LABELS23 ME TO 12_emd_.... """

import pandas as pd
import numpy as np

# from tensorflow.keras.optimizers import SGD

import os

from numpy.random import seed
seed(1)
# from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score



from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold , RepeatedKFold



def emd_names_dic(filename):
    # node_embeddings = pd.read_csv("./Embeddings/")
    lista = os.listdir("./"+filename+"/Embeddings")
    emd_file_names_dic = {i : lista[i] for i in range(len(lista))}
    return emd_file_names_dic


def data(filename,emd_number):
    emd_file_names_dic = emd_names_dic(filename)
    node_embeddings = pd.read_csv("./"+filename+"/Embeddings/"+emd_file_names_dic[emd_number],delimiter=" ",header=None)
    labels =    pd.read_csv("./"+filename+"/"+filename+"_labels.csv",delimiter=" ",header=None)
    
    # return node_embeddings , labels
    
    """ CollegeMsg => 0:1004 , 1:465 , 2:273 ,3:157"""
    """ facebook => 0:1059 , 1:965 , 2:827 ,3:1188"""
    """ Wiki-Vote => 0:5040 , 1:680 , 2:643 ,3:752 """
    
    
    if filename=="CollegeMsg":
        labels = labels[(labels[0] != 0) | (np.random.rand(len(labels)) < .7)] #ποσοστο που κρατας
        # labels = labels[(labels[0] != 1) | (np.random.rand(len(labels)) < .85)]
        node_embeddings2 = pd.concat([node_embeddings,labels],axis=1)
        node_embeddings2 = node_embeddings2.dropna()
        node_embeddings = node_embeddings2.iloc[: , :-1]
    elif filename=="Facebook":
        labels = labels[(labels[0] != 0) | (np.random.rand(len(labels)) < .80)] #ποσοστο που κρατας
        node_embeddings2 = pd.concat([node_embeddings,labels],axis=1)
        node_embeddings2 = node_embeddings2.dropna()
        node_embeddings = node_embeddings2.iloc[: , :-1]
    else:
        labels = labels[(labels[0] != 0) | (np.random.rand(len(labels)) < .70)] #ποσοστο που κρατας
        node_embeddings2 = pd.concat([node_embeddings,labels],axis=1)
        node_embeddings2 = node_embeddings2.dropna()
        node_embeddings = node_embeddings2.iloc[: , :-1]
        
        
    

    
    count = labels[0].groupby(labels[0]).count()
    # print(count)
        
    
    
    return node_embeddings , labels , count


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier


"""=============================================================="""
""" kfold """
"============================================================================================"

from sklearn.metrics import make_scorer , accuracy_score
from sklearn.model_selection import cross_validate



# evaluate a logistic regression model using k-fold cross-validation
from numpy import mean , std
import warnings
warnings.filterwarnings("ignore")
def crossValidation(filename,emd_number):
    node_embeddings , labels , count= data(filename,emd_number)
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
    
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
        
    scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,average = "macro",zero_division=0),
               'recall' : make_scorer(recall_score,average = "macro",zero_division=0), 
               'f1_score' : make_scorer(f1_score,average = "macro",zero_division=0)}
    
    # create model
    model = LogisticRegression(max_iter=1300)
    # evaluate model
    
    result = cross_validate(model, x, y.ravel(), scoring=scoring, cv=cv, n_jobs=-1, error_score="raise")

    
    
    accuracy = float("{:.3f}".format(mean(result["test_accuracy"])))
    precision = float("{:.3f}".format(mean(result["test_precision"])))
    recall = float("{:.3f}".format(mean(result["test_recall"])))
    f1 = float("{:.3f}".format(mean(result["test_f1_score"])))
    # print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

def export_Dataset(filename):
        
    """ CollegeMsg => dic[3] => CollegeMsg_emd_32.csv
        Facebook => dic[2] =>  Facebook_emd_256.csv
        WikiVote => dic[0] => Wiki-Vote_emd_128.csv""" 
    
    if filename=="CollegeMsg":
        node_embeddings , labels , count = data(filename,3)
        result = crossValidation(filename,3)
        print(result)
        dataset = node_embeddings
        dataset[32] = labels[0] #Στην τελευταια στηλη βαζω τα labels
        dataset.to_csv("./"+filename+"/"+filename+"_dataset.csv",index=False,header=False,sep=" ")
    elif filename=="Facebook":
        node_embeddings , labels , count = data(filename,2)
        result = crossValidation(filename,2)
        print(result)
        dataset = node_embeddings
        dataset[256] = labels[0] #Στην τελευταια στηλη βαζω τα labels
        dataset.to_csv("./"+filename+"/"+filename+"_dataset.csv",index=False,header=False,sep=" ")
    else:
        node_embeddings , labels , count = data(filename,0)
        result = crossValidation(filename,0)
        print(result)
        dataset = node_embeddings
        dataset[128] = labels[0] #Στην τελευταια στηλη βαζω τα labels
        dataset.to_csv("./"+filename+"/"+filename+"_dataset.csv",index=False,header=False,sep=" ")
    
    
    
    return dataset

if "__main__":
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
    
    """ CollegeMsg => dic[3] => CollegeMsg_emd_32.csv
        Facebook => dic[2] =>  Facebook_emd_256.csv
        WikiVote => dic[0] => Wiki-Vote_emd_128.csv""" 
    
    # dic = emd_names_dic(filenames[0])
    
    """ KFOLD """
    
    
    # for filename in filenames:
    #     filename = filenames[2]  
    #     print(filename)
    #     print("=========================================")
    #     emd_file_names_dic = emd_names_dic(filename)
    #     for i in range(5):
    #         print("filename =",emd_file_names_dic[i])
    #         accuracy , precision, recall, f1 = crossValidation(filename,i)
           
    #         print(accuracy , precision, recall, f1)
    #         print("")
    #     print("=========================================\n")
    #     break
        
        
        
        
    for filename in filenames:
        print(filename)
        print("=========================================")
       
        dataset = export_Dataset(filename)
        
    