import joblib
import numpy as np

# -----------------------------
# STEP 1: Load trained model
# -----------------------------
model = joblib.load("model/performance_model.pkl")

print("Model loaded successfully!\n")

# -----------------------------
# STEP 2: Take sample input
# -----------------------------
# Format:
# [age, experience, salary, training_hours, attendance, projects_completed]

employee = np.array([[28, 5, 60000, 40, 0.85, 10]])

# -----------------------------
# STEP 3: Make prediction
# -----------------------------
prediction = model.predict(employee)

# -----------------------------
# STEP 4: Show result
# -----------------------------
if prediction[0] == 0:
    print("🔴 Low Performer")
elif prediction[0] == 1:
    print("🟡 Medium Performer")
else:
    print("🟢 High Performer")