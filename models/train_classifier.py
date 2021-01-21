# import libraries
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
#import xgboost as xgb
from sklearn.metrics import classification_report
import numpy as np
import nltk
nltk.download("popular")
import sys
#used to get embeddings
# install latest pip version for this to work
#get_ipython().system('pip3 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.0/en_core_web_sm-2.2.0.tar.gz')
#upgrade numpy and pandas for above -> pip install -U numpy pandas

import spacy 
from sklearn.decomposition import TruncatedSVD
from sklearn.base import BaseEstimator, TransformerMixin
import en_core_web_sm
import pickle

class SpacyVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, nlp):
        self.nlp = nlp
        self.dim = 50

    def fit(self, X, y):
        return self

    def transform(self, X):
        print(X.dtype)
        X.to_csv("temp.csv",index = False)
        # Doc.vector defaults to an average of the token vectors.
        # https://spacy.io/api/doc#vector
        return [self.nlp(text).vector for text in X]
    
 
def load_data(database_filepath):    
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath )
    df = pd.read_sql_table('disaster_response_table', engine ) 
    
    df['message'] = df['message'].astype('str') 
    
    print(df.head())
    # get target variables
    msg_cols = set(["message","id","original","genre"])
    target_cols = set(df.columns).difference(msg_cols)
    X = df["message"]
    Y = df[list(target_cols)]
    return X, Y, Y.columns


def tokenize(text):
    # normalize
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    # tokenize
    tokens = word_tokenize(text) 
    # lemmatizer 
    lemmatizer = nltk.WordNetLemmatizer()
    # remove stop words
    tokens = [ lemmatizer.lemmatize(w) for w in tokens if w not in stop_words ]
    return tokens


def build_model(pipeline="tfidf"):# tfidf or embeddings
    if pipeline == "tfidf":
       pipeline = Pipeline([
            ("vect",CountVectorizer(tokenizer = tokenize)),
            ("tfidf",TfidfTransformer()),
            ("clf",MultiOutputClassifier(RandomForestClassifier()))
        ])
       param_grid = {
        "tfidf__norm":["l1","l2"]
       }

    else:
        nlp = en_core_web_sm.load()
        pipeline = Pipeline(
            steps=[
                ("mean_embeddings", SpacyVectorTransformer(nlp)),
                ("reduce_dim", TruncatedSVD(50)),
                ("clf",MultiOutputClassifier(RandomForestClassifier()))
            ]
        )
        param_grid = {
        "reduce_dim__n_components":[10]
        }
    
    
    # grid search for optimal params
    optimal_model_CV = GridSearchCV(pipeline, param_grid= param_grid) 
    return optimal_model_CV


def plot_scores(y_test, y_pred):
    #Testing the model
    # Printing the classification report for each label
    i = 0
    for col in y_test:
        print('Feature {}: {}'.format(i+1, col))
        print(classification_report(y_test[col], y_pred[:, i]))
        i = i + 1
    #df.values is equivalent to df.to_numpy in pandas 0.24.0+
    accuracy = (y_pred == y_test.values).mean()
    print('The model accuracy is {:.3f}'.format(accuracy))

    
def evaluate_model(model, X_test, Y_test, category_names):
    # Prediction: the multioutput classifier each indivual being Random Forest Classifier  
    # y pred is an array of shape (n_samples, n_classes )
    y_pred = model.predict(X_test)
    plot_scores(Y_test, y_pred)


def save_model(model, model_filepath):
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