
import pandas as pd
import numpy as np

from numpy.random import seed
seed(1)
# from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import KFold , RepeatedKFold

from sklearn.metrics import make_scorer , accuracy_score
from sklearn.model_selection import cross_validate

from sklearn.neighbors import KNeighborsClassifier
    

def read_dataset(filename):
    dataset = pd.read_csv("./"+filename+"/"+filename+"_dataset.csv",delimiter=" ",header=None)
    return dataset

from numpy import mean , std
import warnings
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")


def k_nearest_neighbors(filename , K):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:32]
        labels = dataset[32]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    else:
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
   
    # cv = KFold(n_splits=10, random_state=1, shuffle=True)
    
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=1)
    
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
    print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

import matplotlib.pyplot as plt

def predict(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:32]
        labels = dataset[32]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    else:
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
    # labels
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)
    

    
    # logreg = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=1300)
    model = KNeighborsClassifier(n_neighbors=10)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print('Accuracy of K-Nearest Neighbors classifier on training set: {:.2f}'
          .format(train_score))
    print('Accuracy of K-Nearest Neighbors classifier on test set: {:.2f}'
          .format(test_score))
    print('=======================================')
    predictions = model.predict(x_test)
    
    print(classification_report(y_test,predictions,digits=3))    

def optimal_K(filename):
    """ Minimum error:- 0.3697916666666667 at K = 1 =>  ΜΕ ΤΗΝ ΓΡΑΦΙΚΗ"""
    # dataset = read_dataset(filename)
    # node_embeddings = dataset.iloc[:,0:32]
    # labels = dataset[32]
    
    # x = np.array(node_embeddings.astype(float))
    # y = np.array(labels.astype(float))
    # # labels
    
    # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15)
    # x_train = np.array(x_train)
    # x_test = np.array(x_test)            
    # y_train = np.array(y_train).astype(float)
    # y_test = np.array(y_test).astype(float)    
    
    # error_rate = []
    # for i in range(1,40):
    #     knn = KNeighborsClassifier(n_neighbors=i)
    #     knn.fit(x_train,y_train)
    #     pred_i = knn.predict(x_test)
    #     error_rate.append(np.mean(pred_i != y_test))
    
    # plt.figure(figsize=(10,6))
    # plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
    #      marker='o',markerfacecolor='red', markersize=10)
    # plt.title('Error Rate vs. K Value')
    # plt.xlabel('K')
    # plt.ylabel('Error Rate')
    # print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
    
    """ BEST METRICS FOR K=2 """
    def myFunc(e):
        return -e[1]
    arr = []
    for i in range(1,20):
        accuracy , precision, recall, f1 = k_nearest_neighbors(filename,i)
        arr.append([i,sum([precision,recall,f1])])
        print(accuracy , precision, recall, f1)
    arr = sorted(arr,key=myFunc)
    
    return arr[0][0]
    

    

if "__main__":
    
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
    
    for filename in filenames:
        filename = filenames[2]   

        print(filename)
        print("=========================================")
        K = optimal_K(filename)
        accuracy , precision, recall, f1 = k_nearest_neighbors(filename,K)
        predict(filename)
        break
    
    # K = optimal_K("CollegeMsg")
    # accuracy , precision, recall, f1 = k_nearest_neighbors("CollegeMsg",10)
    # predict("CollegeMsg")