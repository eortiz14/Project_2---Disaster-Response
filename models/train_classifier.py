import sys
import os
import requests
import sqlite3
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

import re
import numpy as np
import pandas as pd

from sklearn.metrics import classification_report, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
import pickle

def load_data(database_filepath):
    
    '''
    load_data extract variables for modelling  from a database
    
    Input:
    database_filepath: Path of database containing clean table 
       
    Output: 
    X: Message text to convert in features
    Y: Dataframe of targets
    '''  
    
    engine = create_engine("sqlite:///" + database_filepath)
    table_name = "DB_Disasters_clean_py"
    df = pd.read_sql_table(table_name, engine)

    # Create X and y variables from the data for modelling
    X = df["message"]
    Y = df.iloc[:, 4:]
    # DROP 'child alone', % of use is 0
    #Y = Y.drop(['child_alone'], axis = 1)
    return X, Y


def tokenize(text):
    
    '''
    tokenize transform a text(message) in words to make it suitable for modelling
    
    Input:
    text: dataframe X['message'] 
       
    Output: 
    lemmed: list of cleaned tokens
    '''
        
    #Remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    
    #Tokenize text
    words = word_tokenize(text)
    
    #Drop stopwords
    words = [w for w in words if w not in stopwords.words("english")]
    
    #Initialize  lemmatizer
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    
    #lemmatize words and convert lowercase
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    
    '''
    Build the pipeline for the text data and define the models for the training,
    the 'features' parts contain TfidfTransformer that vectorizes text and CountVectorizer
    that tokenizes the text
    
    Input:
    None 
       
    Output: 
    pipeline: Model definition
    '''
    
    pipeline = Pipeline([
            ('features', FeatureUnion([
                  ('text_pipeline', Pipeline([
                                               ('vect', CountVectorizer(tokenizer=tokenize)),
                                               ('tfidf', TfidfTransformer()),
                                             ])
                  ),
            ])),
              ("classifier", MultiOutputClassifier(RandomForestClassifier()))
           ])
    parameters = {
            'classifier__estimator__max_depth': [ 8, 16, 32],
            'classifier__estimator__n_estimators': [20, 40, 60]
    }
   
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, col_names):
    
    '''
    evaluates the model with for accuracy, precision and recall metrics
    
    Input:
    Model: Model fit
    X_test: DataFrame of test values
    y_test: DataFrame with the labeled targets
       
    Output: 
    None
    '''
    
    #Model predictions
    Y_pred = pd.DataFrame(model.predict(X_test))
    
    for i, c in enumerate(col_names):

        print(c)

        print(classification_report(Y_test.iloc[:,i],Y_pred.iloc[:,i]))
    
def save_model(model, model_filepath):
    
    '''
    saves the train model
    
    Input:
    model: fitted model
    model_filepath: path for saving fitted model
       
    Output: 
    None
    '''
    
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y = load_data(database_filepath)
        col_names = Y.columns.tolist()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, col_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()