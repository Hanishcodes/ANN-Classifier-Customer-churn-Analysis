import pandas as pd
import numpy as np
import streamlit as st
import tensorflow  as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import pickle

## load model
model=load_model('model.h5')

### load one hot encoder and label encoder,standard scaler
with open("one_hot_encoder_geo.pkl",'rb') as file:
    one_hot_encoder_geo=pickle.load(file)
with open("lbl_encoder.pkl",'rb') as file:
    ldl_encoder=pickle.load(file)
with open("scaler.pkl",'rb') as file:
    scaler=pickle.load(file)



## use streamlit framework
st.title("Customber Churn Analysis")

credit_score=st.number_input('CreditScore')
geography=st.selectbox('Geography',one_hot_encoder_geo.categories_[0])
gender=st.selectbox('Gender',ldl_encoder.classes_)
age=st.slider('Age',15,80)
tenure=st.slider('Tenure',1,12)
balance=st.number_input('Balance')
num_of_products=st.slider('NumOfProducts',1,4)
has_cr_card=st.selectbox('HasCrCard',[0,1])
is_active_member=st.selectbox('IsActiveMember',[0,1])
estimated_salary=st.number_input('EstimatedSalary')


### take one input
input_data=pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender':[ldl_encoder.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})


encoded_geo=one_hot_encoder_geo.transform([[geography]]).toarray()
encoded_df_geo=pd.DataFrame(encoded_geo,columns=one_hot_encoder_geo.get_feature_names_out(['Geography']))

## concatenate input_data and encoded_df_geo dataframe
input_data=pd.concat([input_data.reset_index(drop=True),encoded_df_geo],axis=1)


## scale the input data
input_data_scaled=scaler.transform(input_data)


## prediction
prediction=model.predict(input_data_scaled)
prediction_prob=prediction[0][0]
st.write(f'Churn Probability: {prediction_prob:.2f}')

if prediction_prob>0.5:
    st.write("customer likely to churn")
else:
    st.write("customer is not likely to churn")

