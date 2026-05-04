import pandas as pd

df = pd.read_csv("../data/raw/METABRIC_RNA_Mutation.csv")

# Drop rows with missing label
df = df.dropna(subset=["overall_survival"])

# Convert label to int
df["overall_survival"] = df["overall_survival"].astype(int)

# Select numeric columns only (auto-removes strings)
numeric_df = df.select_dtypes(include=["int64", "float64"])

# Remove label from features
X = numeric_df.drop(columns=["overall_survival", "patient_id"], errors="ignore")
y = df["overall_survival"]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFeature preview:")
print(X.head())

print("\nLabel distribution:")
print(y.value_counts())