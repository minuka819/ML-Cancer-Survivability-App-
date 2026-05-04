import pandas as pd

df = pd.read_csv("../data/raw/METABRIC_RNA_Mutation.csv")

print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nHead:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nPotential label columns (low unique values):")
for col in df.columns:
    if df[col].nunique() < 10:
        print(col, "->", df[col].unique())