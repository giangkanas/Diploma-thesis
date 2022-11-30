
import pandas as pd
import numpy as np

from numpy.random import seed
seed(1)
# from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import KFold 

from sklearn.metrics import make_scorer , accuracy_score
from sklearn.model_selection import cross_validate

""" MODELS """
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.svm import SVC     
""" NEURAL NETWORK """ 
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 

import matplotlib.pyplot as plt


def read_dataset(filename):
    dataset = pd.read_csv("./"+filename+"/"+filename+"_dataset.csv",delimiter=" ",header=None)
    return dataset

import warnings
from numpy import mean

warnings.filterwarnings("ignore")


def naiveBayes(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
   
    
    cv = KFold(n_splits=10, random_state=1, shuffle=True)    
    scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,average = "macro",zero_division=0),
               'recall' : make_scorer(recall_score,average = "macro",zero_division=0), 
               'f1_score' : make_scorer(f1_score,average = "macro",zero_division=0)}
    
    # create model
    model = GaussianNB()
    # evaluate model
    # scores = cross_val_score(model, x, y, scoring='precision', cv=cv, n_jobs=-1)
    result = cross_validate(model, x, y.ravel(), scoring=scoring, cv=cv, n_jobs=-1, error_score="raise")
    # report performance

    accuracy = float("{:.3f}".format(mean(result["test_accuracy"])))
    precision = float("{:.3f}".format(mean(result["test_precision"])))
    recall = float("{:.3f}".format(mean(result["test_recall"])))
    f1 = float("{:.3f}".format(mean(result["test_f1_score"])))
    # print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

def decisionTree(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
   
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    
    
    scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,average = "macro",zero_division=0),
               'recall' : make_scorer(recall_score,average = "macro",zero_division=0), 
               'f1_score' : make_scorer(f1_score,average = "macro",zero_division=0)}
    
    # create model
    model = DecisionTreeClassifier()
    # evaluate model
    # scores = cross_val_score(model, x, y, scoring='precision', cv=cv, n_jobs=-1)
    result = cross_validate(model, x, y.ravel(), scoring=scoring, cv=cv, n_jobs=-1, error_score="raise")
    # report performance

    
    
    accuracy = float("{:.3f}".format(mean(result["test_accuracy"])))
    precision = float("{:.3f}".format(mean(result["test_precision"])))
    recall = float("{:.3f}".format(mean(result["test_recall"])))
    f1 = float("{:.3f}".format(mean(result["test_f1_score"])))
    # print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

def k_nearest_neighbors(filename , K):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
   
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    
    
    scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,average = "macro",zero_division=0),
               'recall' : make_scorer(recall_score,average = "macro",zero_division=0), 
               'f1_score' : make_scorer(f1_score,average = "macro",zero_division=0)}
    
    # create model
    model = KNeighborsClassifier(n_neighbors=K)
    # evaluate model
    # scores = cross_val_score(model, x, y, scoring='precision', cv=cv, n_jobs=-1)
    result = cross_validate(model, x, y.ravel(), scoring=scoring, cv=cv, n_jobs=-1, error_score="raise")
    # report performance

    
    
    accuracy = float("{:.3f}".format(mean(result["test_accuracy"])))
    precision = float("{:.3f}".format(mean(result["test_precision"])))
    recall = float("{:.3f}".format(mean(result["test_recall"])))
    f1 = float("{:.3f}".format(mean(result["test_f1_score"])))
    # print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1    

def optimal_K(filename):
    
    def myFunc(e):
        return -e[1]
    arr = []
    for i in range(1,11):
        accuracy , precision, recall, f1 = k_nearest_neighbors(filename,i)
        arr.append([i,sum([precision,recall,f1])])
    arr = sorted(arr,key=myFunc)
    
    return arr[0][0]

def logisticRegression(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
    
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
    # scores = cross_val_score(model, x, y, scoring='precision', cv=cv, n_jobs=-1)
    result = cross_validate(model, x, y.ravel(), scoring=scoring, cv=cv, n_jobs=-1, error_score="raise")
    # report performance

    
    
    accuracy = float("{:.3f}".format(mean(result["test_accuracy"])))
    precision = float("{:.3f}".format(mean(result["test_precision"])))
    recall = float("{:.3f}".format(mean(result["test_recall"])))
    f1 = float("{:.3f}".format(mean(result["test_f1_score"])))
    # print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

def svm(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
   
    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    
    
    
    
    scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,average = "macro",zero_division=0),
               'recall' : make_scorer(recall_score,average = "macro",zero_division=0), 
               'f1_score' : make_scorer(f1_score,average = "macro",zero_division=0)}
    
    # create model
    model = SVC()
    
    # evaluate model
    # scores = cross_val_score(model, x, y, scoring='precision', cv=cv, n_jobs=-1)
    result = cross_validate(model, x, y.ravel(), scoring=scoring, cv=cv, n_jobs=-1, error_score="raise")
    # report performance

    
    
    accuracy = float("{:.3f}".format(mean(result["test_accuracy"])))
    precision = float("{:.3f}".format(mean(result["test_precision"])))
    recall = float("{:.3f}".format(mean(result["test_recall"])))
    f1 = float("{:.3f}".format(mean(result["test_f1_score"])))
    # print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

""" ================NEURAL NETWORK==================== """

def baseline_model1():
    """ ΜΟΝΤΕΛΟ ΠΟΥ ΠΡΟΕΚΥΨΕ ΑΠΟ HYPERBAND """
    model = Sequential([
        Dense(296, input_dim = 256, activation='relu'),
        Dense(units=104, activation='relu'),
        Dense(units=72, activation='relu'),
        Dense(units=4, activation='softmax')])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
    
    return model

def baseline_model2():
    """ ΜΟΝΤΕΛΟ ΠΟΥ ΠΡΟΕΚΥΨΕ ΑΠΟ HYPERBAND """
    model = Sequential([
        Dense(480, input_dim = 128, activation='relu'),
        Dense(units=24, activation='relu'),
        Dense(units=56, activation='relu'),
        Dense(units=4, activation='softmax')])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
    
    return model

def baseline_model3():
    """ ΜΟΝΤΕΛΟ ΠΟΥ ΠΡΟΕΚΥΨΕ ΑΠΟ HYPERBAND """
    model = Sequential([
        Dense(224, input_dim = 64, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=72, activation='relu'),
        Dense(units=4, activation='softmax')])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
    
    return model
    


def neural_network(filename):
    
     
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
        batch_size = 128
        build_fn=baseline_model1
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
        batch_size = 512
        build_fn=baseline_model2
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
        batch_size = 512
        build_fn=baseline_model3
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
     

 


    """CollegeMsg =>best batch size = 128 """
    """Facebook =>best batch size = 512 """
    """WikiVotes =>best batch size = 512 """


    estimator = KerasClassifier(build_fn=build_fn, epochs=50, batch_size=batch_size, verbose=1 )
    
    
    cv = KFold(n_splits=10, shuffle=True)
   
    
    scoring = {'accuracy' : make_scorer(accuracy_score), 
               'precision' : make_scorer(precision_score,average = "macro",zero_division=0),
               'recall' : make_scorer(recall_score,average = "macro",zero_division=0), 
               'f1_score' : make_scorer(f1_score,average = "macro",zero_division=0)}
    
    
    result = cross_validate(estimator, x, y.ravel(), scoring=scoring, cv=cv, n_jobs=-1, error_score="raise")
    
    accuracy = float("{:.3f}".format(mean(result["test_accuracy"])))
    precision = float("{:.3f}".format(mean(result["test_precision"])))
    recall = float("{:.3f}".format(mean(result["test_recall"])))
    f1 = float("{:.3f}".format(mean(result["test_f1_score"])))
    # print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

""" =================================================== """

def compute_metrics(filename):
    metrics = []
    
    accuracy , precision, recall, f1 = naiveBayes(filename)
    metrics.append([accuracy , precision, recall, f1])
    print("Bayes:",accuracy , precision, recall, f1)
    
    
    accuracy , precision, recall, f1 = decisionTree(filename)
    metrics.append([accuracy , precision, recall, f1])
    print("decisionTree:",accuracy , precision, recall, f1)
    
    K = optimal_K(filename)
    accuracy , precision, recall, f1 = k_nearest_neighbors(filename,K)
    metrics.append([accuracy , precision, recall, f1])
    print("k_nearest_neighbors: k=>",K,accuracy , precision, recall, f1)
    
    accuracy , precision, recall, f1 = logisticRegression(filename)
    metrics.append([accuracy , precision, recall, f1])
    print("logisticRegression:",accuracy , precision, recall, f1)
    
    accuracy , precision, recall, f1 = svm(filename)
    metrics.append([accuracy , precision, recall, f1])
    print("svm:",accuracy , precision, recall, f1)
    
    accuracy , precision, recall, f1 = neural_network(filename)
    metrics.append([accuracy , precision, recall, f1])
    print("neural_network:",accuracy , precision, recall, f1)
    
    classifiers = ["GaussianNB" , "Decision Tree" , "K-nearest neighbors",
                   "Logistic Regression" , "Svm" , "Neural Network"]
    metrics = pd.DataFrame(metrics, columns = ['accuracy' , 'precision' , 'recall' , 'f1'], index=classifiers)
    
    metrics.to_csv("./"+filename+"/"+filename+"_metrics.csv" , sep=" ", header=None )
    return metrics

def plot_bargrams(filename):
    metrics = compute_metrics(filename)
    # metrics = pd.read_csv("./"+filename+"/"+filename+"_metrics.csv" , sep=" ", index_col=0,header=None)
    # metrics = metrics.rename(columns = {1:"accuracy",2:"precision",3:"recall",4:"f1"})
    
    
    fig = metrics.plot.bar(
        rot=0 ,
        # figsize=(20,15) , 
        width = 0.5,
        
        title = filename + " Metrics Comparison").get_figure()
               
    plt.xticks(rotation=45, horizontalalignment="center")
    
    plt.show()
    fig.savefig(
        './'+filename+'/'+filename+"_metrics.png")

if "__main__":
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
    

    
    for filename in filenames:
        # filename = filenames[2]
        print(filename)
        print("=========================================")
        plot_bargrams(filename)
        
        
        print()
        # break
        
    
