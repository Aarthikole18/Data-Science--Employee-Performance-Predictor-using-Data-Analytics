import streamlit as st
import numpy as np
import joblib

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("model/performance_model.pkl")

# -----------------------------
# UI TITLE
# -----------------------------
st.title("💼 Employee Performance Predictor")

st.write("Enter employee details to predict performance level")

# -----------------------------
# INPUT FIELDS
# -----------------------------
age = st.slider("Age", 20, 60)
experience = st.slider("Experience (Years)", 0, 35)
salary = st.number_input("Salary", min_value=10000)
training_hours = st.slider("Training Hours", 0, 100)
attendance = st.slider("Attendance", 0.0, 1.0)
projects = st.slider("Projects Completed", 0, 20)

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Performance"):

    input_data = np.array([[age, experience, salary, training_hours, attendance, projects]])
    
    prediction = model.predict(input_data)

    # -----------------------------
    # RESULT DISPLAY
    # -----------------------------
    if prediction[0] == 0:
        st.error("🔴 Low Performer")
    elif prediction[0] == 1:
        st.warning("🟡 Medium Performer")
    else:
        st.success("🟢 High Performer")