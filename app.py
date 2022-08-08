import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import requests
import base64
from fpdf import FPDF
import calendar
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import random
from tensorflow.keras.optimizers import SGD, Adam
from keras.models import Sequential
from sklearn.metrics import confusion_matrix,classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Flatten,  BatchNormalization, Conv1D,MaxPooling1D



#Define all the Sections
header = st.container()
cred = st.container()
rawdata = st.container()
pprocess = st.container()
ousage = st.container()
eda = st.container()
ddis = st.container()
dprep = st.container()
ttsplit = st.container()
dmodel = st.container()
tmodel = st.container()
emodel = st.container()
pmodel = st.container()

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

		
		
with ousage:
	st.subheader('Overall Usage of Customers')
	col1, col2 = st.columns(2)
	with col1:
		sns.set_theme(style="whitegrid")
		fig1 = plt.figure(figsize=(10,10))
		sns.barplot(x=bad_f['Hr'], y=bad_f['Usage'], hue=bad_f['Year']).set(title='Bad Customer Overall Usage')
		st.pyplot(fig1)
	with col2:
		sns.set_theme(style="whitegrid")
		fig2 = plt.figure(figsize=(10,10))
		sns.barplot(x=gooddf['Hr'], y=gooddf['Usage'], hue=gooddf['Year']).set(title='Good Customer Overall Usage')
		st.pyplot(fig2)
			


with eda:
	
	st.subheader('Data Visuals')
	datav = st.selectbox('At what level Do you want the Usage report?',('Season','Weekdays','Month','Year'), index=0, help='Select Visualization')	
	col5, col6=st.columns(2)
	if datav == 'Season':	
		with col5:
			st.text('Seasonal Usage Bad Plot')
			sns.set_theme(style="whitegrid")
			fig3 = plt.figure(figsize=(10,10))
			sns.lineplot(x=bad_f['Hr'], y=bad_f['Usage'], hue=bad_f['Season']).set(title='Bad Customer Seasonal Usage')
			st.pyplot(fig3)
		with col6:
			st.text('Seasonal Usage Good Plot')
			sns.set_theme(style="whitegrid")
			fig4 = plt.figure(figsize=(10,10))
			sns.lineplot(x=gooddf['Hr'], y=gooddf['Usage'], hue=gooddf['Season']).set(title='Good Customer Seasonal Usage')
			st.pyplot(fig4)
	elif datav =='Weekdays':	
		with col5:
			st.text('Daily Usage Plot Bad Customer')
			sns.set_theme(style="whitegrid")
			fig5 = plt.figure(figsize=(10,10))
			sns.lineplot(x=bad_f['Hr'], y=bad_f['Usage'], hue=bad_f['Dayname']).set(title='Bad Customer Daily Usage')
			st.pyplot(fig5)
			#st.text('Seasonal Plot')
		with col6:
			st.text('Daily Usage Plot Good Customer ')
			sns.set_theme(style="whitegrid")
			fig6 = plt.figure(figsize=(10,10))
			sns.lineplot(x=gooddf['Hr'], y=gooddf['Usage'], hue=gooddf['Dayname']).set(title='Good Customer Daily Usage')
			st.pyplot(fig6)
			#st.text('Seasonal Plot')
	elif datav =='Month':
		with col5:
			st.text('Monthly Usage Plot Bad Customer')
			sns.set_theme(style="whitegrid")
			fig7 = plt.figure(figsize=(10,10))
			sns.lineplot(x=bad_f['Hr'], y=bad_f['Usage'], hue=bad_f['Month']).set(title='Bad Residential Monthly Usage')
			st.pyplot(fig7)
		with col6:
			st.text('Monthly Usage Plot Good Customer')
			sns.set_theme(style="whitegrid")
			fig8 = plt.figure(figsize=(10,10))
			sns.lineplot(x=gooddf['Hr'], y=gooddf['Usage'], hue=gooddf['Month']).set(title='Good Residential Monthly Usage')
			st.pyplot(fig8)
	elif datav =='Year':
		with col5:
			st.text('Annual Usage Plot Bad Customer')
			sns.set_theme(style="whitegrid")
			fig9 = plt.figure(figsize=(10,10))
			sns.lineplot(x=bad_f['Hr'], y=bad_f['Usage'], hue=bad_f['Year']).set(title='Bad Residential Annual Usage')
			st.pyplot(fig9)
		with col6:
			st.text('Annual Usage Plot Good Customer')
			sns.set_theme(style="whitegrid")
			fig10 = plt.figure(figsize=(10,10))
			sns.lineplot(x=gooddf['Hr'], y=gooddf['Usage'], hue=gooddf['Year']).set(title='Good Residential Annual Usage')
			st.pyplot(fig10)
with ddis:
	
	st.subheader('Usage Data Distribution')
	col7, col8 = st.columns(2)
	with col7:
		st.text('Usage Distribution Bad Customer')
		fig11 = plt.figure(figsize=(10,10))
		sns.histplot(data=bad_f, x=bad_f['Usage'],binwidth=5, kde=True)
		st.pyplot(fig11)
	with col8:
		st.text('Usage Distribution Good Customer')
		fig12 = plt.figure(figsize=(10,10))
		sns.histplot(data=gooddf, x=gooddf['Usage'], binwidth=5,kde=True)
		st.pyplot(fig12)
		
with dprep:
	
	st.subheader('Data Preparation')
	GR_Con = gooddf.pivot_table(index=('Meter','Hr'), columns='Date', values='Usage',aggfunc='sum')
	BR_Con = bad_f.pivot_table(index=('Meter','Hr'), columns='Date', values='Usage',aggfunc='sum')
	st.text('Pivoted Data')
	col9, col10 = st.columns(2)
	col9.write(BR_Con)
	col10.write(GR_Con)
	# Start getting the input format
	x=[]
	n=1 # Class for Good Customer
	for i in gooddf['Meter'].unique():
  		y=GR_Con.loc[i,:].transpose().to_numpy()
  		x.append([y,n])
	w=[]
	t=0 # Class for Bad Customer
	for j in bad_f['Meter'].unique():
		z=BR_Con.loc[j,:].transpose().to_numpy()
		w.append([z,t])
	st.text('Final Input data for Model')
	col9.write(w[0])
	col10.write(x[0])
	Usage_list = x+w
			
with ttsplit:
	
	st.subheader('Train-Test Split')
	col11, col12 = st.columns(2)
	test = col11.slider('Select testing data (in %)', min_value=0.1, max_value=0.5, value=0.1, step=0.1, help='Select the test data percentage by sliding the slider ')
	train= col12.metric(label ='Train data (in %): ', value=1-test)
	def SplitFeaturesAndLabels(data):
    		X = []
    		y = []
    		for features, labels in data: #Label are the Categories available in the data array
        		X.append(features)
        		y.append(labels)
    		X = np.array(X) #Features Vector
    		y = np.array(y) #Label Vector
    		return X, y
	
	X, y = SplitFeaturesAndLabels(Usage_list)
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test, random_state=42)
	col11.write('Shape of Training data post data split')
	col11.write(X_train.shape)
	col12.write('Shape of Training Label post split')	
	col12.write(y_train.shape)

with dmodel:
	st.subheader('Define Convolution Model Hyper Parameters')
	inp_shape= X_train.shape[1:]
	N_lable =2 # Total number of classes
	col13, col14 = st.columns(2)
	lrt = col13.slider('Select Learning Rate', min_value=0.01, max_value=0.05, value=0.01, step=0.001, help='Select the test data percentage by sliding the slider ')
	opti = col14.radio('Select Model Optimizer',('SGD','Adam'))
	if opti == 'SGD':
		opt=SGD(learning_rate=lrt)
	elif opti =='Adam':
		opt=Adam(learning_rate=lrt)
	model =Sequential()
	model.add(Conv1D(filters=64,kernel_size=3,strides =1,activation='relu',input_shape=inp_shape))
	model.add(Conv1D(filters=32,kernel_size=3,activation='relu'))
	model.add(BatchNormalization())
	model.add(MaxPooling1D())
	model.add(Dense(100, activation='sigmoid'))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Dense(N_lable, activation="softmax"))
	model.compile(loss="mae",optimizer=opt,metrics=['accuracy'])
	model.summary()
	
with tmodel:
	st.subheader('Model Training')
	epch =st.slider('Select Epochs', min_value=10, max_value=1000, value=10, step=10, help='Select the training epoch size')
	Model_final = model.fit(X_train, y_train, batch_size=20,epochs=epch, verbose=0)
	fig13 = plt.figure(figsize=(10,10))
	plt.plot(Model_final.history["accuracy"])
	plt.xlabel("Epoch #")
	plt.ylabel("Accuracy")
	plt.title("Training Accuracy")
	st.pyplot(fig13)

with emodel:
	st.subheader('Model Evaluation')
	Actual_Test =[]
	Pred_Test =[]
	Pred = model.predict(X_test)
	for i in range(len(X_test)):
  		Actual_Test.append(y_test[i])
  		Pred_Test.append(np.argmax(Pred[i]))
	result =confusion_matrix(Actual_Test,Pred_Test)
	
	fig14,ax = plt.subplots(figsize=(10,10))
	sns.heatmap(result, annot=True, ax=ax)
	ax.set_xlabel('Predicted Class')
	ax.set_ylabel('Actual Class')
	st.pyplot(fig14)
	report = classification_report(y_test, Pred_Test)
	st.write(report)
	#st.write(result)
	#Predhat = model.predict_classes(X_test, verbose=0)
	#Pred = Pred[:, 0]
	#Predhat = Predhat[:, 0]
	
with pmodel:
	st.subheader('Predict Customer Type')
	gcl = gooddf['Meter'].unique().tolist()
	bcl = bad_f['Meter'].unique().tolist()
	cl = gcl+bcl
	datap = st.selectbox('Select Meter to Predict',cl, index=0, help='Select Meter for which Prediction needs to be made')
	compc = gooddf.append(bad_f, ignore_index=True)
	compm = compc.loc[compc['Meter'] == datap][['Meter','Date','Hr','Usage']]
	st.dataframe(compm)
	compm = compm.pivot_table(index=('Meter','Hr'), columns='Date', values='Usage',aggfunc='sum')
	xp=compm.transpose().to_numpy()
	xp= np.array(xp)
	st.write(xp.shape)
	xp =xp.reshape(1, 536,24)
	st.write(xp.shape)
	pred_op = model.predict(xp)
	
	st.write((pred_op))

	
	
	
