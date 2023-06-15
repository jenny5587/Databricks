# Databricks notebook source
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Abalone dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
df = pd.read_csv(url, names=column_names)

# Calculate summary statistics
summary = df.describe()
print("Summary Statistics:")
print(summary)

# Visualize the data
# Pairwise scatter plot
sns.pairplot(df)
plt.title("Pairwise Scatter Plot")
plt.show()

# Correlation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
