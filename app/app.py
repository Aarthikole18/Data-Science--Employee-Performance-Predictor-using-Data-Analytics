import streamlit as st
import numpy as np
import joblib
import os

# -----------------------------
# LOAD MODEL (SAFE PATH)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(BASE_DIR, "model", "performance_model.pkl")

model = joblib.load(model_path)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Employee Performance Predictor", page_icon="💼")

# -----------------------------
# TITLE
# -----------------------------
st.title("💼 Employee Performance Predictor")
st.markdown("### 🚀 Predict employee performance using Machine Learning")

st.write("Fill the details below to get prediction")

# -----------------------------
# INPUT FIELDS
# -----------------------------
age = st.slider("Age", 20, 60, 30)
experience = st.slider("Experience (Years)", 0, 35, 5)
salary = st.number_input("Salary (₹)", min_value=10000, value=50000)
training_hours = st.slider("Training Hours", 0, 100, 40)
attendance = st.slider("Attendance (0 to 1)", 0.0, 1.0, 0.85)
projects = st.slider("Projects Completed", 0, 20, 10)

# -----------------------------
# PREDICTION BUTTON
# -----------------------------
if st.button("🔮 Predict Performance"):

    input_data = np.array([[age, experience, salary, training_hours, attendance, projects]])

    prediction = model.predict(input_data)

    # -----------------------------
    # RESULT DISPLAY
    # -----------------------------
    st.subheader("📊 Prediction Result")

    if prediction[0] == 0:
        st.error("🔴 Low Performer")
    elif prediction[0] == 1:
        st.warning("🟡 Medium Performer")
    else:
        st.success("🟢 High Performer")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("💡 Built using Machine Learning + Streamlit")