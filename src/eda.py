import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data/employee_data.csv")

print("Dataset Preview:")
print(df.head())

# -----------------------------
# BASIC INFO
# -----------------------------
print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

# -----------------------------
# PERFORMANCE DISTRIBUTION
# -----------------------------
plt.figure()
sns.countplot(x="performance", data=df)
plt.title("Employee Performance Distribution")
plt.show()

# -----------------------------
# CORRELATION HEATMAP
# -----------------------------
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# -----------------------------
# RELATIONSHIP ANALYSIS
# -----------------------------
plt.figure()
sns.boxplot(x="performance", y="attendance", data=df)
plt.title("Attendance vs Performance")
plt.show()

plt.figure()
sns.boxplot(x="performance", y="training_hours", data=df)
plt.title("Training Hours vs Performance")
plt.show()

plt.figure()
sns.boxplot(x="performance", y="projects_completed", data=df)
plt.title("Projects vs Performance")
plt.show()

print("EDA completed successfully!")