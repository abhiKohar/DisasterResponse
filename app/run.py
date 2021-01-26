import json
import plotly
import pandas as pd
import re

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords  

import os
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

NUM_WORDS = 30

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    stop_words = set(stopwords.words('english'))  
  
      

    clean_tokens = [w for w in clean_tokens if not w in stop_words]  

    return clean_tokens

# load data
engine = create_engine('sqlite:////home/workspace/data/DisasterResponse.db')
df = pd.read_sql_table('disaster_response_table', engine)

# load model
model = joblib.load("/home/workspace/models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # modify and extract data for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    df_copy = df.drop(["id","genre","original","message"],axis =1)
    
    cols = []
    x0 = []
    x1 = []
    for column in list(df_copy.columns):
        cols.append(column)
        x0.append(df[column].value_counts()[0])
        #print (df[column].value_counts())
        x1.append(df[column].value_counts()[1])
        
    count_words = {}
    
    for sent in df["message"]:
        tokens = tokenize(sent)
        for w in tokens:
            if re.findall('[^A-Za-z0-9]',w): # string has special characters ignore it
                continue;
            if w in count_words:
                count_words[w] += 1
            else:
                count_words[w] = 1
    
    # sort dict by values
    count_words = sorted(count_words.items(), key=lambda item: item[1])
#     print (count_words)
    
    word_labels = []
    word_freq = []
    
    for i in range (1,NUM_WORDS):
        word_labels.append(count_words[-i][0])
    for i in range (1,NUM_WORDS):
        word_freq.append(count_words[-i][1])
    
    
    # create visuals
    
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cols,
                    y=x0,
                    name = "Label 0"
                    
                ),
                Bar(
                    x=cols,
                    y=x1,
                    name = "Label 1"
                )
            ],

            'layout': {
                'title': 'Distribution of Target Labels',
                'yaxis': {
                    'title': "Count of Labels"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=word_labels,
                    y=word_freq
                )
            ],

            'layout': {
                'title': 'Distribution of Words in Messages',
                'yaxis': {
                    'title': "Word Frequency"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    
    
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()