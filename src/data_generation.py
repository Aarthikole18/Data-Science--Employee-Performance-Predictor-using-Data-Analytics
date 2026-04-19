import pandas as pd
import numpy as np

# -----------------------------
# STEP 1: Set random seed
# -----------------------------
np.random.seed(42)

# -----------------------------
# STEP 2: Create fake employees
# -----------------------------
n = 1000

df = pd.DataFrame({
    "age": np.random.randint(22, 60, n),
    "experience": np.random.randint(0, 35, n),
    "salary": np.random.randint(30000, 150000, n),
    "training_hours": np.random.randint(0, 100, n),
    "attendance": np.random.uniform(0.5, 1.0, n),
    "projects_completed": np.random.randint(1, 20, n)
})

# -----------------------------
# STEP 3: Create performance logic
# -----------------------------
df["performance_score"] = (
    df["attendance"] * 0.4 +
    df["projects_completed"] * 0.4 +
    df["training_hours"] * 0.2
)

# -----------------------------
# STEP 4: Convert score into labels
# -----------------------------
df["performance"] = pd.cut(
    df["performance_score"],
    bins=3,
    labels=["Low", "Medium", "High"]
)

# -----------------------------
# STEP 5: Save dataset
# -----------------------------
df.to_csv("data/employee_data.csv", index=False)

print("✅ Dataset created successfully!")
print(df.head())