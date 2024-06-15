#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import dump
from joblib import load
import streamlit as st
from streamlit_jupyter import StreamlitPatcher, tqdm
StreamlitPatcher().jupyter()


# In[2]:


model = load('Accident_severity.joblib')


# In[3]:


# Create a Streamlit app
st.title("Accident Severity Prediction")

# Input fields for feature values on the main screen
st.header("Enter accident attributes")
days_of_week = st.selectbox("Days of week", ('Friday','Thursday','Wednesday','Tuesday','Monday','Saturday','Sunday'))
casualty_sex = st.selectbox("Casualty Sex", ('Male', 'Female', 'Na'))
light = st.selectbox("Light Conditions", ('Daylight', 'Darkness - lights Lit','Darkness - lights UnLit','Darkness - No lights',))
casualty_age_band = st.number_input("Casualty age", min_value=0, max_value=200, value=50)
number_of_vehicles = st.number_input("Vehicles invloved", min_value=0, max_value=10000, value=0)

# Map input values to numeric using the label mapping
label_mapping = {'Friday':0,'Thursday':4,'Wednesday':6,'Tuesday':5,'Monday':1,'Saturday':2,'Sunday':3,
                'Male':1,'Female':0,'Na':2,
                'Daylight':3,'Darkness - lights Lit':0,'Darkness - No lights':2,'Darkness - lights UnLit':1}

days_of_week = label_mapping[days_of_week]
casualty_sex = label_mapping[casualty_sex]
light = label_mapping[light]

# Make a prediction using the model
prediction = model.predict([[days_of_week, casualty_sex, light, casualty_age_band, number_of_vehicles]])

# Display the prediction result on the main screen
st.header("Prediction Result")
if prediction[0] == 0:
    st.success("This person is slightly injured.")
elif prediction[0] == 1:
    st.error("This person is seriously injured.")
else:
    st.error("This person has a fatal injury.")

