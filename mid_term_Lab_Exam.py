import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DATA_FILE = "Titanic-Dataset.csv"

# Load and inspect the dataset
df = pd.read_csv(DATA_FILE)

print("First five rows:")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nColumn information:")
df.info()
print("\nSummary statistics:")
print(df.describe())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Clean columns needed for the analysis
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df = df.drop(columns=["Cabin"])

print("\nOverall survival rate:", round(df["Survived"].mean() * 100, 2), "%")
print("\nSurvival rate by gender:")
print(df.groupby("Sex")["Survived"].mean().round(3))
print("\nSurvival rate by passenger class:")
print(df.groupby("Pclass")["Survived"].mean().round(3))

# Visual exploration
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(16, 9))

sns.countplot(data=df, x="Survived", ax=axes[0, 0])
axes[0, 0].set_title("Survival Count")
axes[0, 0].set_xticks([0, 1], ["Did not survive", "Survived"])

sns.countplot(data=df, x="Sex", hue="Survived", ax=axes[0, 1])
axes[0, 1].set_title("Survival by Gender")

sns.countplot(data=df, x="Pclass", hue="Survived", ax=axes[0, 2])
axes[0, 2].set_title("Survival by Passenger Class")

sns.histplot(data=df, x="Age", bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Age Distribution")

sns.boxplot(data=df, x="Pclass", y="Fare", ax=axes[1, 1])
axes[1, 1].set_title("Fare by Passenger Class")

numeric_columns = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]
sns.heatmap(
    df[numeric_columns].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    ax=axes[1, 2],
)
axes[1, 2].set_title("Correlation Heatmap")

plt.tight_layout()
plt.show()
