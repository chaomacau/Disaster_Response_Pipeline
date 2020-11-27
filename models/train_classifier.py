import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    '''
    Load database 
    
    Args: 
        database_filepath - path to database
        
    Return:
        X - Features dataframe
        Y - Labels dataframe
        category_names - category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('Messages_Data',con=engine)

    X = df['message']
    Y = df[df.columns[5:]]
    category_names = Y.columns
    return X,Y, category_names


def tokenize(text):
    '''
    Tokenize text by normalization, stop words removal, and stemming and lemmatization
    
    Args:
        text - messages
        
    Return:
        lemmed - tokenized words
    '''
    ##Normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    ##Stop words Removal
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    words = [w for w in tokens if w not in stop_words]
    ##Stemming and Lemmatization
    stemmer = PorterStemmer()
    stemmed = [stemmer.stem(w) for w in words]
    lemmatizer = WordNetLemmatizer()
    lemmed = [lemmatizer.lemmatize(w) for w in stemmed]
    return(lemmed)


def build_model():
    '''
    Build a pipeline as well we GridSearchCV model
    
    Return:
        cv - pipeline with grid searched parameters
    '''
    pipeline = Pipeline(
    [('vect', CountVectorizer(tokenizer=tokenize)), 
     ('tfidf', TfidfTransformer()), 
     ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))),]
    )   
    
    parameters = {
    'vect__ngram_range': ((1,1),(1,2)),
    #'vect__max_df': (0.5, 0.75, 1.0),
    #'vect__max_features': (None, 5000,10000),
    #'tfidf__use_idf': (True, False),
    #'clf__estimator__n_estimators': [50, 100, 200],
    #'clf__estimator_min_samples_split': [2, 3, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate model by returning classification reports
    
    Args:
        model - Model built
        X_test - messages
        Y_test - Classification result
        category_names - category names
        
    Return:
        printing of classification reports
    '''
    y_pred=model.predict(X_test)
    print(classification_report(Y_test, y_pred, target_names=category_names))
    

def save_model(model, model_filepath):
    '''
    Save model
    
    Args: 
        model - model to be saved
        model_filepath - filepath to save model
        
    Return:
        None
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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