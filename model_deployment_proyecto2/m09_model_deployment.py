#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer


def split_into_lemmas(text):
    # Lematizacion
    wordnet_lemmatizer = WordNetLemmatizer()
    #nltk.download('wordnet')
    text = text.lower()
    words = text.split()
    return [wordnet_lemmatizer.lemmatize(word, pos='v') for word in words]

def clasificar_pelicula(Year, Title, Plot):

    clf = joblib.load(os.path.dirname(__file__) + '/clasificacion_peliculas.pkl') 
    vect = joblib.load(os.path.dirname(__file__) + '/vectorizer.pkl')

    data = [[Year, Title, Plot]]
    df_params = pd.DataFrame(data, columns=['Year', 'Title', 'Plot'])
    
    X_test_dtm = vect.transform(df_params['Title']+' '+df_params['Plot'])
    dfXTestTemp = pd.DataFrame(X_test_dtm.toarray(), index= df_params.index)
    dfXTestFinal = df_params.join(dfXTestTemp)
    dfXTestFinal = dfXTestFinal.drop('Title', 1)
    dfXTestFinal = dfXTestFinal.drop('Plot', 1)
    
    y_pred_test_genres = clf.predict_proba(dfXTestFinal)
    
    cols = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
        'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
        'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']
    
    res = pd.DataFrame(y_pred_test_genres, index=df_params.index, columns=cols)

    index_top_ = np.argsort(-res.values, axis=1)[:, :5]
    ret = ''
    
    for x in index_top_:
        list_top = x.tolist()
        for y in list_top:
            ret = ret + str(res.columns[y]) + ': ' + str( "{:.2f}".format(res.at[0,res.columns[y]])) + ', '
    
    ret = ret[:-2]

    return ret

if __name__ == "__main__":
    
    if len(sys.argv) < 4:
        print('Please add all parameters (Year, Title, Plot)')
        
    else:

        y = sys.argv[1]
        t = sys.argv[2]
        p = sys.argv[3]


        p1 = clasificar_pelicula(y,t,p)
        
        print('This movie can be classified as: ', p1)
        