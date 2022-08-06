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

with header:
	font="sans serif"
	textColor="#26273"
	st.title('Electricity Theft Prediction')

with cred:
	st.subheader('Login')
	col1, col2 = st.columns(2)
	owner = col1.text_input('Git Owner Name', value='', help='Enter User Id')
	token = col2.text_input('Git Private Token', value='', help='Enter Password')
	if owner != 'sample':
		st.write('Wrong User Name')
		st.stop()
	if token != 'test123':
		st.write('Wrong Token')
		st.stop()
	st.write('Credentials Correct')
