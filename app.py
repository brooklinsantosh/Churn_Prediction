import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pickle

#Load the model
model = tf.keras.models.load_model('artifacts/models/model.h5')

# Load all the pickle files
with open('artifacts/models/le_gender.pkl','rb') as file:
    le_gender = pickle.load(file)

with open('artifacts/models/ohe_geo.pkl','rb') as file:
    ohe_geo = pickle.load(file)

with open('artifacts/models/scaler.pkl','rb') as file:
    scaler = pickle.load(file) 

#Stramlit app
st.title('Customer Churn Prediction')  

#User inputs
geography =  st.selectbox('Geography', ohe_geo.categories_[0])
gender = st.selectbox('Gender', le_gender.classes_)
age = st.slider('Age',18,92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure',0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card', [0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

#Prepare input data
input_df = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [le_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = ohe_geo.transform(input_df[['Geography']])
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geo.get_feature_names_out(['Geography']))

input_df = pd.concat([input_df.drop('Geography',axis=1), geo_encoded_df], axis=1)

input_data_scaled = scaler.transform(input_df)

pred = model.predict(input_data_scaled)[0][0]

st.write(f'Churn probability: {pred:.2f}')

if pred>0.5:
    st.write('The customer is likely to churn')
else:
    st.write('The customer is not likely to churn')