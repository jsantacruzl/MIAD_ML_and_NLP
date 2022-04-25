#!/usr/bin/python

import pandas as pd
import joblib
import sys
import os

def predict_price(Year, Mileage, State, Make, Model):

    # Import pkls
    clf = joblib.load(os.path.dirname(__file__) + '/pronostico_precios.pkl') 
    cbe_encoder = joblib.load(os.path.dirname(__file__) + '/cbe_encoder.pkl')

    # Create features
    data = [[Year, Mileage, State, Make, Model]]
    df_params = pd.DataFrame(data, columns=['Year', 'Mileage','State','Make','Model'])
    YTotalEncoded = cbe_encoder.transform(df_params)
    
    # Make prediction
    p1 = clf.predict(YTotalEncoded)[0]

    return p1


if __name__ == "__main__":
    
    if len(sys.argv) < 6:
        print('Please add all parameters (Year, Mileage, State, Make, Model)')
        
    else:

        y = sys.argv[1]
        m = sys.argv[2]
        s = sys.argv[3]
        ma = sys.argv[4]
        o = sys.argv[5]

        p1 = predict_price(y,m,s,ma,o)
        
        print('Price Estimated: ', p1)
        