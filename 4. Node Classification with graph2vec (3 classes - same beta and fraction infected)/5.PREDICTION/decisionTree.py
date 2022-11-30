
import pandas as pd
import numpy as np

from numpy.random import seed
seed(1)
# from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

from sklearn.model_selection import KFold , RepeatedKFold

from sklearn.metrics import make_scorer , accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
    

def read_dataset(filename):
    dataset = pd.read_csv("./"+filename+"/"+filename+"_dataset.csv",delimiter=" ",header=None)
    return dataset

from numpy import mean , std
import warnings
from sklearn.model_selection import train_test_split


warnings.filterwarnings("ignore")
def decisionTree(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:16]
        labels = dataset[16]
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
    model = DecisionTreeClassifier()
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

def predict(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:16]
        labels = dataset[16]
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
    

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
          .format(train_score))
    print('Accuracy of Decision Tree classifier on test set: {:.2f}'
          .format(test_score))
    print('=======================================')
    predictions = model.predict(x_test)
    
    print(classification_report(y_test,predictions,digits=3))
    

if "__main__":
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
    
    for filename in filenames:
        filename = filenames[2]   
        print(filenames[2])
        print("=========================================")
        accuracy , precision, recall, f1 = decisionTree(filenames[2])
        predict(filenames[2])
        break
    
    # predict("CollegeMsg")