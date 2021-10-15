# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
st.title("Dream Homes.com")
st.text('Find your dream home here')
df=pd.read_csv('kc_house_data.csv')
st.image('house.jpg')
#p_range=st.slider("Price Range",min_value=15000,max_value=1000000,step=1,value=15000)
#st.text('Price selected is Rs' + str(p_range))
p_range = st.slider("Price Range ", min_value=int(df['price'].min()),max_value=int(df['price'].max()),step=1,value=int(df['price'].min()))
st.text('Price selected is Rs ' + str(p_range))
fig=px.scatter_mapbox(df.loc[df['price']<p_range],lat='lat',lon='long',color='sqft_living',size='price')
fig.update_layout(mapbox_style='open_street_map')
st.plotly_chart(fig)
st.header("Price Predictor")
sel_box_var=st.selectbox("Select Method",['Linear','Ridge','Lasso'],index=0)
multi_var=st.multiselect("Select Additional Variables for accuracy =",['sqft_living','sqft_lot','sqft_basement'])
df_new=[]
df_new=df[multi_var]
if sel_box_var =='Linear':
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    x=df_new
    y=df['price']
    model=LinearRegression()
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R square="+str(r2_score(Y_test,Y_pred)))
    st.text("MSE="+str(mean_squared_error(Y_test,Y_pred)))
elif sel_box_var =='Lasso':
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    x=df_new
    y=df['price']
    model=Lasso()
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R square="+str(r2_score(Y_test,Y_pred)))
    st.text("MSE="+str(mean_squared_error(Y_test,Y_pred)))
else:
    df_new['bedrooms']=df['bedrooms']
    df_new['bathrooms']=df['bathrooms']
    x=df_new
    y=df['price']
    model=Ridge()
    X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2)
    reg=model.fit(X_train,Y_train)
    Y_pred=reg.predict(X_test)
    st.text("Intercept="+str(reg.intercept_))
    st.text("Coefficient="+str(reg.coef_))
    st.text("R square="+str(r2_score(Y_test,Y_pred)))
    st.text("MSE="+str(mean_squared_error(Y_test,Y_pred)))
#st.set_option('depreciation.showPyplotGlobalUse',False)
st.set_option('deprecation.showPyplotGlobalUse', False)
sns.regplot(Y_test,Y_pred)
st.pyplot()
count=0
predicted_val=0
for i in df_new.keys():
    try:
        val=st.text_input("ENter no./val off",+i)
        predicted_val=float(val)*reg.coef_[count]+predicted_val
        count=count+1
    except:
        pass
st.text('Predicted Pricess are:'+str(predicted_val+reg.intercept_))
st.header("Application details")
img=st.file_uploader("Upload Application")
st.text("Details for the representative to contact you ")
st.text("Enter your address")
address=st.text_area("Your address Here")
date=st.date_input("Enter a date")
time=st.time_input("Enter the time")
if st.checkbox("I confirm the date and time", value = False):
    st.write("Thanks for confirming")
st.number_input("Rate our site",min_value = 1, max_value = 10, step=1)

    