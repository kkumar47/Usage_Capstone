import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import requests
import calendar
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
import random
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten,  BatchNormalization, Conv1D,MaxPooling1D



#Define all the Sections
header = st.container()
cred = st.container()
rawdata = st.container()
pprocess = st.container()
ousage = st.container()

with header:
	font="sans serif"
	textColor="#26273"
	st.title('Electricity Theft Prediction')

with cred:
	st.subheader('Login')
	col1, col2 = st.columns(2)
	owner = col1.text_input('User Name', value='', help='Enter User Id')
	token = col2.text_input('Password', value='', help='Enter Password')
	if owner != 'sample':
		st.write('Wrong User Name')
		st.stop()
	if token != 'test123':
		st.write('Wrong Token')
		st.stop()
	st.write('Credentials Correct')
def raw_data():
	return pd.read_csv('https://raw.githubusercontent.com/kkumar47/Usage_Capstone/master/Raw.txt.txt')
def good_cust():
	return pd.read_csv('https://raw.githubusercontent.com/kkumar47/Usage_Capstone/master/Fl_Good_Customer.csv')
def bad_cust():
	return pd.read_csv('https://raw.githubusercontent.com/kkumar47/Usage_Capstone/master/Fl_Bad_Customer.csv')

rawdf = raw_data()
baddf = bad_cust()
gooddf = good_cust()

with rawdata:
	st.subheader("Electricity Usage History data for Customers", anchor ='The Data')
	st.text('Raw Data')
	#The raw data is displayed here
	st.dataframe(rawdf.head(10))
	rawd = rawdf.to_csv().encode('utf-8')
	st.download_button('Download Data', data=rawd, file_name='Raw Data.csv', help='Download Data in CSV format')
	
with pprocess:
	st.subheader("Preprocess Dataset")
	pbutton = st.button('Start Process', help='Click to generate additional fields')
	if pbutton == True:
		st.write('Processing Data please wait....')
		@st.cache
		def cleanse(baddf):
			baddf['Date'] = baddf['Day'].apply(lambda x: pd.to_datetime((2009*1000 )+ x, format = "%Y%j") if x<=365 else pd.to_datetime((2010*1000 )+ (x-365), format = "%Y%j"))
			baddf['Day_Num'] = baddf['Date'].apply(lambda x: x.weekday())
			baddf['Dayname'] = baddf['Date'].apply(lambda x: calendar.day_name[x.weekday()])
			baddf['Holiday_Ind'] = baddf['Day_Num'].apply(lambda x: 0 if x<=4 else 1)
			baddf['Month'] = baddf['Date'].apply(lambda x: x.strftime("%B"))
			baddf['Year'] = baddf['Date'].apply(lambda x: x.year)
			def condition(x):
  				if (x=='August' or x=='September' or x=='October'):
    					return "Autumn"
  				elif (x=='November' or x=='December' or x=='January'):
    					return "Winter"
  				elif (x=='February' or x=='March' or x=='April'):
    					return "Spring"
  				elif (x=='May' or x=='June' or x=='July'):
    					return "Summer"
			baddf['Season']=baddf['Month'].apply(condition)
			baddf.drop(baddf[baddf['Hr']==24].index, inplace=True)
			baddf.dropna(axis=0,inplace=True)
			return baddf
		
		bad_f =cleanse(baddf)
		st.text('Processed Bad Customer')
		st.dataframe(bad_f)
		bad_fd = bad_f.to_csv().encode('utf-8')
		st.download_button('Download Data', data=bad_fd, file_name='Bad Customer Data.csv', help='Download Data in CSV format')
		st.text('Processed Good Customer')
		gooddf.drop(gooddf[gooddf['Hr']==24].index, inplace=True)
		gooddf.dropna(axis=0,inplace=True)
		st.dataframe(gooddf)
		good_dfd = gooddf.to_csv().encode('utf-8')
		st.download_button('Download Data', data=good_dfd, file_name='Good Customer Data.csv', help='Download Data in CSV format')
	else:
		st.write('Click Start Process to continue')
		
with ousage:
	st.subheader('Overall Usage of Customers')
	col1, col2 = st.columns(2)
	with col1:
		sns.set_theme(style="whitegrid")
		fig1 = plt.figure(figsize=(10,10))
		sns.barplot(x=gooddf['Hr'], y=gooddf['Usage'], hue=gooddf['Year']).set(title='Good Customer Overall Usage')
		st.pyplot(fig1)
	with col2:
		sns.set_theme(style="whitegrid")
		fig2 = plt.figure(figsize=(10,10))
		sns.barplot(x=bad_f['Hr'], y=bad_f['Usage'], hue=bad_f['Year']).set(title='Bad Customer Overall Usage')
		st.pyplot(fig2)
		
