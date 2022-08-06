import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
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
