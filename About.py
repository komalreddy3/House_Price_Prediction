import streamlit as st
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df1=pd.read_csv(open('cols.csv'))
csvfile=open('output.csv','rb')

df=pd.read_csv(csvfile)
X=df.drop(['price'],axis='columns')
y=df['price']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
lr_clf = LinearRegression()
lr_clf.fit(X_train,y_train)
lr_clf.score(X_test,y_test)

def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]
st.title('Home price prediction')
 
st.write('---')
 
# area of the house
sqft = st.slider('Area of the house', 1000, 5000, 1500)
 
# no. of bedrooms in the house
bhk = st.number_input('No. of bedrooms', min_value=0, step=1)
 
# no. of bathrooms in the house
bath = st.number_input('No. of bathrooms', min_value=0, step=1)

location = st.selectbox("Pick Location ",df1.columns[1:])
 
if st.button('Predict House Price'):
    cost = predict_price(location,sqft, bath,bhk)
    st.text(cost)
