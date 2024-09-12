import streamlit as st
import numpy as np
import pickle
import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load the vectorizer and model
vectorizer = joblib.load(open("tfidf_vectorizer.pkl", 'rb'))

model = pickle.load(open("logistic_regression_model (1).pkl", 'rb'))

# Streamlit UI
st.markdown(
    """
    <h1 style='text-align: center; color: darkblue;'>Railway Emergency Text Classification</h1>
    """,
    unsafe_allow_html=True)

PNR_no = st.text_input("Enter PNR No.")
complaint = st.text_input("Enter your complaint:")
submit = st.button("Classify")

if submit:
    if complaint:
        complaint_transformed = vectorizer.transform([complaint])
        print("Transformed input:", complaint_transformed.toarray())


        start = time.time()
        prediction = model.predict(complaint_transformed)
        end = time.time()

        st.write('Prediction time taken:', round(end - start, 2), 'seconds')

        # Since prediction is already a string
        if prediction[0] == 'Emergency Complaint':
            st.markdown(
                """
                <h3 style=' font-size: 20px;'>
                 <span style='color: black;'>Provided complaint is a</span>
                 <span style=' font-size: 20px; color: red;'>Emergency Complaint</span>
                </h3>
                """,
                unsafe_allow_html=True
            )
        elif prediction[0] == 'Common Complaint':
            st.markdown(
                """
                <h3 style=' font-size: 20px;'>
                 <span style='color: black;'>Provided complaint is a</span>
                 <span style='color: green;'>Common Complaint</span>
                </h3>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <h3 style=' font-size: 20px;'>
                 <span style='color: black;'>An unexpected error has occurred</span>
                </h3>
                """,
                unsafe_allow_html=True
            )
    else:
        st.markdown(
            """
        <h3 style='text-align: center; font-size: 24px;color: red;'>Please enter your complaint</h3>
        """,
            unsafe_allow_html=True
        )


