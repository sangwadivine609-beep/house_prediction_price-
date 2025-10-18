# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 09:48:29 2025

@author: user
"""

import numpy as np
import streamlit as st
import pickle

# Load the trained model
with open('house_sales_data.sav', 'rb') as file:
    model = pickle.load(file)

# Streamlit UI
st.title("🏠 House Price Regression Prediction")
st.write("Fill in the details below to predict the estimated house price:")

# User Inputs
area = st.number_input("area", value=2014)
rooms = st.number_input("rooms", value=3)



# Prediction on button click
if st.button("Predict"):
    # Create input in correct order and shape
    input_data = np.array([[area,rooms]])

    # Predict using loaded model
    prediction = model.predict(input_data)

    # Show result

    st.success(f"💰 Predicted House Price: ${prediction[0]:,.2f}")
