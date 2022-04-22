import streamlit as st
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from PIL import Image
warnings.filterwarnings('ignore')


global df  
#df = pd.read_csv("icu2.0.csv")
#df.tail(50)
#st.write(df)
#st.write(df.shape)
#st.write(df.describe())
#chart=st.line_chart(df)

st.title("Patient Monitoring in Intensive Care Unit")
image = Image.open(os.path.join('C:\\Users\\Philip\\Desktop\\icu ml C\\1.jpg')) 
st.image(image)


#uploaded_file = st.file_uploader("Choose a file")
#if uploaded_file is not None:

  #df = pd.read_csv(uploaded_file)

df = pd.read_csv('C:\\Users\\Philip\\Desktop\\icu ml C\\ICU csv.csv')
  
  #st.subheader('DATA INFORMATION')
  #st.write(df)


#st.subheader('DATA SHAPE')
#st.write(df.shape)
#fig, ax = plt.subplots()
#df.hist(
    #bins=8,
    #column="Heart Rate",
    #grid=False,
    #figsize=(8, 8),
    #color="#86bf91",
    #zorder=2,
    #rwidth=0.9,
    #ax=ax,
#)
def get_user_input():
    age =st.sidebar.slider('Age',0,60,40)
    hb =st.sidebar.slider('Heart Rate',0,200,90)
    pressure =st.sidebar.slider('Pressure',0,250,120)
    temperature  =st.sidebar.slider('Temperature',0,45,36)
    x_axis =st.sidebar.slider('x_axis',0,400,300)
    y_axis =st.sidebar.slider('y_axis',0,400,250)
    z_axis =st.sidebar.slider('z_axis',0,400,300)
    user_data ={'Age': age,
                'Heart Rate':hb,
                'Pressure':pressure,
                'Temperature':temperature,
                'x_axis':x_axis,
                'y_axis':y_axis,
                'z_axis':z_axis,
    }
    features=pd.DataFrame(user_data,index=[0])
    return features
user_input =get_user_input()
st.write(user_input)

#st.subheader('PLOTTED HISTOGRAM GRAPH')
#st.write(fig)
#fig1, ax1 = plt.subplots()
#df.hist(
 #   bins=8,
  #  column="Pressure",
   # grid=False,
    #figsize=(8, 8),
    #color="#86bf91",
    #zorder=2,
    #rwidth=0.9,
    #ax=ax1,
#)
#st.write(fig1)
#fig2, ax2 = plt.subplots()
#df.hist(
 #   bins=8,
  #  column="Temperature",
   # grid=False,
    #figsize=(8, 8),
    #color="#86bf91",
    #zorder=2,
    #rwidth=0.9,
    #ax=ax2,
#)
#st.write(fig2)
#st.subheader('DESCRIBE METHOD')
#st.write(df.describe())
#st.subheader('DATA PREPROCESSING')
#st.write(df.isnull().sum())

from sklearn.preprocessing import Normalizer
norm=Normalizer().fit(df)
normalization=norm.transform(df)
normalization=pd.DataFrame(normalization,index=df.index,columns=df.columns)

#st.subheader('EQUAL WEIGHTAGE')
#st.write(normalization)

#x =normalization[['Age','Heart Rate','Pressure','Temperature','x_axis','y_axis','z_axis']]
x =df[['Age','Heart Rate','Pressure','Temperature','x_axis','y_axis','z_axis']]
y =df[['label']]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.3,random_state=42)
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x_train, y_train)
dt_confidence = regressor.score(x_test, y_test)
#st.write("dt accuracy: ", dt_confidence)
prediction= regressor.predict(user_input)
st.write(prediction)

if prediction == 0:
   st.write("Normal")
   st.write("Can be discharged in 1 Day")
elif prediction == 1:
   st.write("Neutral")
   st.write("Can be discharged in 2 Days")
else:
   st.write("Abnormal")
   st.write("Needs Intensive care")
