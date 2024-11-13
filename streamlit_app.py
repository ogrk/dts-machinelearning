import streamlit as st
import pandas as pd

st.title('Diabetes Prediction App')

st.info('This app uses a machine learning model to predict diabetes')

df = pd.read_csv('https://raw.githubusercontent.com/ogrk/data/refs/heads/main/clean_data.csv')
