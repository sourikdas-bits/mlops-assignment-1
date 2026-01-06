#!/usr/bin/env python


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
data = pd.read_csv("./data/heart.csv")

# Clean and preprocess
# Check for missing values
print("Missing values per column:\n", data.isnull().sum())

# If missing values exist, fill or drop (example: fill with median)
data = data.fillna(data.median(numeric_only=True))

# print out the first few rows to verify
print("First few rows of the dataset:\n", data.head())

# print descriptive statistics
print("Descriptive statistics:\n", data.describe())

# Encode categorical features if any
categorical_cols = data.select_dtypes(include=['object', 'category']).columns
if len(categorical_cols) > 0:
    data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# EDA

# 1. Class balance
plt.figure(figsize=(6,4))
sns.countplot(x='target', hue='target', data=data, palette='Set2', legend=False)
plt.title('Class Balance')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("../output/class_balance.png")
plt.close()

# 2. Histograms for all features
data.hist(figsize=(16,12), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("../output/feature_histograms.png")
plt.close()

# 3. Correlation heatmap
plt.figure(figsize=(12,10))
corr = data.corr()
sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig("../output/correlation_heatmap.png")
plt.close()

print("EDA visualizations saved: class_balance.png, feature_histograms.png, correlation_heatmap.png")
