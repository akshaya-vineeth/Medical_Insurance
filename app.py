import streamlit as st
from src.prediction import Insurance_Prediction


st.title('Insurance Predictor System')

st.write('This is a Insurance Predcitor System used to predict your Insurance amount')

Age = st.number_input('Enter Your Age')
Annual_income = st.number_input('Enter Your Annual Income in LPA')
term_years = st.number_input('Enter Your policy Term')
sum_assured = st.number_input('Enter Sum assurance in Lakhs')

if st.button('Predict'):
    model = Insurance_Prediction()
    result = model.predict(Age,Annual_income,term_years,sum_assured)
    st.success(result)