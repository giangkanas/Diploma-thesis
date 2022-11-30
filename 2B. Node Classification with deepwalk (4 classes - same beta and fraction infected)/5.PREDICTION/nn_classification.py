# import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Flatten
import tensorflow.keras.optimizers as opt
import tensorflow.keras.callbacks as callbacks
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import SGD

import os

from numpy.random import seed
seed(1)
# from tensorflow.keras import backend as K
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold,RepeatedKFold


from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
def read_dataset(filename):
    dataset = pd.read_csv("./"+filename+"/"+filename+"_dataset.csv",delimiter=" ",header=None)
    return dataset




from sklearn.metrics import make_scorer , accuracy_score
from sklearn.model_selection import cross_validate
from numpy import mean

"============================================================================================"
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier

import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import make_scorer 
from sklearn.metrics import precision_score, recall_score, f1_score

from keras_tuner.tuners import Hyperband
def build_model(hp):
    model = Sequential()
    # Tune the number of units in the first Dense layer
    # Choose an optimal value between 32-512
    hp_units1 = hp.Int('units1', min_value=8, max_value=512, step=8)
    hp_units2 = hp.Int('units2', min_value=0, max_value=128, step=8)
    hp_units3 = hp.Int('units3', min_value=0, max_value=128, step=8)
    model.add(Dense(units=hp_units1, activation='relu'))
    model.add(Dense(units=hp_units2, activation='relu'))
    model.add(Dense(units=hp_units3, activation='relu'))
    model.add(Dense(4, kernel_initializer='normal', activation='softmax'))
      
    # Tune the learning rate for the optimizer
    # Choose an optimal value from 0.01, 0.001, or 0.0001

    

  
  
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
      
    return model

def HyperBand(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:16]
        labels = dataset[16]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    
    x_train, x_test, y_train, y_test = train_test_split(node_embeddings,labels,test_size= 0.2)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    
    best_models = []
    batch_sizes = [64,128,256,512]
    for batch_size in batch_sizes:
        tuner = Hyperband(
            build_model,
            max_epochs=50,
            executions_per_trial=5,
            objective='val_accuracy',
            # directory="Hyperband1"
            directory="./"+filename+"/Hyperband1"
            
        )
        
        callback = EarlyStopping(monitor='val_loss', mode="min",patience=3)
        tuner.search(x_train,y_train,epochs=50,validation_data=(x_test,y_test),callbacks=[callback], batch_size=batch_size)
        
        best_model = tuner.get_best_models()[0]
        best_model.build(x_train.shape)
        best_models.append(best_model)
    
        best_model.summary()
        
    
        callback = EarlyStopping(monitor='val_loss', mode="min",patience=3)
        history = best_models[0].fit(x_train,y_train,epochs = 50,validation_data=(x_test, y_test), batch_size=batch_size, callbacks = [callback])
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
        print("\n\n\n================\n\n\n")
        
    """CollegeMsg =>best batch size = 256 """
    """Facebook =>best batch size = 64 """
    """WikiVotes =>best batch size = 256 """
    
def baseline_model1():
    """ ΜΟΝΤΕΛΟ ΠΟΥ ΠΡΟΕΚΥΨΕ ΑΠΟ HYPERBAND """
    model = Sequential([
        Dense(216, input_dim = 16, activation='relu'),
        Dense(units=72, activation='relu'),
        Dense(units=48, activation='relu'),
        Dense(units=4, activation='softmax')])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
    
    return model

def baseline_model2():
    """ ΜΟΝΤΕΛΟ ΠΟΥ ΠΡΟΕΚΥΨΕ ΑΠΟ HYPERBAND """
    model = Sequential([
        Dense(504, input_dim = 128, activation='relu'),
        Dense(units=24, activation='relu'),
        Dense(units=16, activation='relu'),
        Dense(units=4, activation='softmax')])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
    
    return model

def baseline_model3():
    """ ΜΟΝΤΕΛΟ ΠΟΥ ΠΡΟΕΚΥΨΕ ΑΠΟ HYPERBAND """
    model = Sequential([
        Dense(504, input_dim = 256, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=120, activation='relu'),
        Dense(units=4, activation='softmax')])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
    
    return model
    


def crossValidation(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:16]
        labels = dataset[16]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    
    x = np.array(node_embeddings.astype(float))
    y = np.array(labels.astype(float))
     


    """CollegeMsg =>best batch size = 256 """
    """Facebook =>best batch size = 64 """
    """WikiVotes =>best batch size = 256 """
    estimator = KerasClassifier(build_fn=baseline_model3, epochs=50, batch_size=256, verbose=1 )
    
    # cv = RepeatedKFold(n_splits=10, n_repeats=4, random_state=1)
    
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
    print(accuracy , precision, recall, f1)
        
    return accuracy , precision, recall, f1

def predictInfluence(filename):
    dataset = read_dataset(filename)
    if filename=="CollegeMsg":
        node_embeddings = dataset.iloc[:,0:16]
        labels = dataset[16]
    elif filename == "Facebook":
        node_embeddings = dataset.iloc[:,0:128]
        labels = dataset[128]
    else:
        node_embeddings = dataset.iloc[:,0:256]
        labels = dataset[256]
    

    x_train, x_test, y_train, y_test = train_test_split(node_embeddings,labels,test_size= 0.15)
    x_train = np.array(x_train)
    x_test = np.array(x_test)            
    y_train = np.array(y_train).astype(float)
    y_test = np.array(y_test).astype(float)
    
    
    """ MODEL """
    model = Sequential([
        Dense(504, input_dim = 256, activation='relu'),
        Dense(units=32, activation='relu'),
        Dense(units=120, activation='relu'),
        Dense(units=4, activation='softmax')])
        
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"] )
        
    
    # optimizer = opt.SGD(lr=1e-02)
    optimizer = opt.Adam()
    callback = EarlyStopping(monitor='val_loss', mode="min",patience=3)      
    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"] )
    history = model.fit(x_train,y_train,epochs=50,batch_size=256,validation_data=(x_test, y_test),callbacks= callback)
    
    """EVALUATE"""
    loss , accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    labels = y_test
    # predictions = model.predict_classes(x_test) """ στην εκδοση 2.8 που εχω εχει καταργηθει και εχει αντικατασταθει """
    predictions=model.predict(x_test) 
    predictions=np.argmax(predictions,axis=1)

    # return y_test , predictions
    """ METRICS """
    precision = precision_score(y_test,  predictions, average='macro')
    recall = recall_score(y_test,  predictions, average='macro')
    f1 = f1_score(y_test,  predictions, average='macro')
    
    print("\nMetrics\n ")
    print('Loss: %.2f' % (loss))
    print('Accuracy: %.2f' % (accuracy*100))
    print('Precision: %.2f' % (precision*100))
    print('Recall: %.2f' % (recall*100))
    print('F1_score: %.2f' % (f1*100))
    
    """ GRAPH FOR LOSS """
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()
    
    
    print(classification_report(y_test,predictions,digits=3))
    


if "__main__":
    
    filenames = ["CollegeMsg","Facebook","Wiki-Vote"]
   
    for filename in filenames:
        filename = filenames[2]
        print(filename)
        print("=========================================")
        # HyperBand(filename)
        accuracy , precision, recall, f1 = crossValidation(filename)
        predictInfluence(filename)
        
       
        break
    
    




