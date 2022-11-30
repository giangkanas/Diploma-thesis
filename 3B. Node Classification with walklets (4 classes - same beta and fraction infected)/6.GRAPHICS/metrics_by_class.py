
import pandas as pd
import numpy as np

from numpy.random import seed
seed(1)
# from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score , classification_report , confusion_matrix

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
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import seaborn as sns

def read_dataset(filename):
    dataset = pd.read_csv("./"+filename+"/"+filename+"_dataset.csv",delimiter=" ",header=None)
    return dataset

import warnings
from numpy import mean

warnings.filterwarnings("ignore")


# from sklearn.metrics import multilabel_confusion_matrix

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
    # labels
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)
    

    # logreg = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=1300)
    model = GaussianNB()
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

        


    print('Accuracy of Naive Bayes classifier on training set: {:.2f}'
          .format(train_score))
    print('Accuracy of Naive Bayes classifier on test set: {:.2f}'
          .format(test_score))
    print('=======================================')
    predictions = model.predict(x_test)
    
    
    return y_test , predictions
        

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
    
    return y_test , predictions

def k_nearest_neighbors(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
        k=1
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
        k=1
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
        k=1
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
    # labels
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)
    

    
    # logreg = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=1300)
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)
    print('Accuracy of K-Nearest Neighbors classifier on training set: {:.2f}'
          .format(train_score))
    print('Accuracy of K-Nearest Neighbors classifier on test set: {:.2f}'
          .format(test_score))
    print('=======================================')
    predictions = model.predict(x_test)
    
    return y_test , predictions



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
    # labels
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)
    

    """ LOGISTIC REGRESSION """
    # logreg = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=1300)
    logreg = LogisticRegression(max_iter=1300)
    logreg.fit(x_train, y_train)
    train_score = logreg.score(x_train, y_train)
    test_score = logreg.score(x_test, y_test)

    
    print('Accuracy of Logistic regression classifier on training set: {:.2f}'
          .format(train_score))
    print('Accuracy of Logistic regression classifier on test set: {:.2f}'
          .format(test_score))
    print('=======================================')
    predictions = logreg.predict(x_test)
    
    return y_test , predictions

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
    # labels
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size= 0.15)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)
    

    # logreg = LogisticRegressionCV(Cs=10, cv=10, scoring="accuracy", verbose=False, multi_class="ovr", max_iter=1300)
    model = SVC()
    model.fit(x_train, y_train)
    train_score = model.score(x_train, y_train)
    test_score = model.score(x_test, y_test)

    
    print('Accuracy of SVM classifier on training set: {:.2f}'
          .format(train_score))
    print('Accuracy of SVM classifier on test set: {:.2f}'
          .format(test_score))
    print('=======================================')
    predictions = model.predict(x_test)
    
    return y_test , predictions

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
        model=baseline_model1()
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
        batch_size = 512
        model=baseline_model2()
    else:
        node_embeddings = dataset.iloc[:,0:64]
        labels = dataset[64]
        batch_size = 512
        model=baseline_model3()
    
    
    

    x_train, x_test, y_train, y_test = train_test_split(node_embeddings,labels,test_size= 0.15)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)
    
    
    
    
    # optimizer = opt.SGD(lr=1e-02)

    callback = EarlyStopping(monitor='val_loss', mode="min",patience=3)      
    
    model.fit(x_train,y_train,epochs=50,batch_size=batch_size,validation_data=(x_test, y_test),callbacks= callback)
    
    """EVALUATE"""
    loss , accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    labels = y_test
    # predictions = model.predict_classes(x_test) """ στην εκδοση 2.8 που εχω εχει καταργηθει και εχει αντικατασταθει """
    predictions=model.predict(x_test) 
    predictions=np.argmax(predictions,axis=1)

     

    
    
    return y_test , predictions

""" =================================================== """

def plot_matrix(y_test , predictions, methodName, filename):
    report = classification_report(y_test,predictions,digits=3, output_dict=True)

    report = pd.DataFrame(report).transpose()
    report = report[["precision","recall","f1-score"]]
    
    report = report.reset_index()
    report = report[["precision","recall","f1-score"]]
    report = report.head(4) # 4 classes
    plt.figure(figsize=(5,4))
    sns.heatmap(report, annot=True)
    plt.title(methodName)
    
    path = './'+filename
    if not os.path.exists(path+'/Matrixes'):
        os.makedirs(path+'/Matrixes')
    
    plt.savefig(path+'/Matrixes/'+methodName+"_PRF.png")
    plt.show()
    
    
    """ CONFUSION MATRIX"""
    cf_matrix = confusion_matrix(y_test,predictions)
    cm_df = pd.DataFrame(cf_matrix,
                     index = [0,1,2,3], 
                     columns = [0,1,2,3])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True)
    plt.title(methodName+' Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    
    plt.savefig(path+'/Matrixes/'+methodName+"confusion_matrix.png")
    plt.show()

def compute_matrix(filename):

    
    y_test, predictions = naiveBayes(filename)
    plot_matrix(y_test , predictions , "GaussianNB" , filename)
    
    
    y_test, predictions = decisionTree(filename)
    plot_matrix(y_test , predictions , "Decision Tree", filename)
    
    y_test, predictions = k_nearest_neighbors(filename)
    plot_matrix(y_test , predictions , "K-nearest neighbors", filename)
    
    y_test, predictions = logisticRegression(filename)
    plot_matrix(y_test , predictions , "Logistic Regression", filename)
    
    y_test, predictions = svm(filename)
    plot_matrix(y_test , predictions , "Svm", filename)
    
    y_test, predictions = neural_network(filename)
    plot_matrix(y_test , predictions , "Neural Network", filename)
    
    

def plot_bargrams(filename):
    # metrics = compute_metrics(filename)
    metrics = pd.read_csv("./"+filename+"/"+filename+"_metrics.csv" , sep=" ", index_col=0,header=None)
    metrics = metrics.rename(columns = {1:"accuracy",2:"precision",3:"recall",4:"f1"})
    
    
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
        
        compute_matrix(filename)
        
        print()
        # break
        
    
