import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -----------------------------
# STEP 1: Load dataset
# -----------------------------
df = pd.read_csv("data/employee_data.csv")

print("Dataset Loaded")

# -----------------------------
# STEP 2: Encode target column
# -----------------------------
le = LabelEncoder()
df["performance"] = le.fit_transform(df["performance"])

# Mapping:
# Low = 0
# Medium = 1
# High = 2

# -----------------------------
# STEP 3: Split features & target
# -----------------------------
X = df.drop(["performance", "performance_score"], axis=1)
y = df["performance"]

# -----------------------------
# STEP 4: Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# STEP 5: Train model
# -----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -----------------------------
# STEP 6: Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# STEP 7: Evaluation
# -----------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# STEP 8: Save model
# -----------------------------
joblib.dump(model, "model/performance_model.pkl")

print("\nModel saved successfully!")