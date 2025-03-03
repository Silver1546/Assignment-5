# app.py

import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import datasets

# Load Dataset and Scaler
iris = datasets.load_iris()
scaler = StandardScaler()
scaler.fit(iris.data)

# Load Model
with open('iris_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Iris Flower Species Prediction")
st.write("Enter petal and sepal dimensions to predict the species.")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.write(f"Predicted Species: {iris.target_names[prediction[0]]}")