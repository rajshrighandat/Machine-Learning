# 1. Library Import
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 2. Read Data
df = pd.read_csv("EV_Adoption.csv")

# 3. EDA
print(df.head())
df.info()
print(df.isnull().sum())

# Handle missing values before encoding and model training
numeric_columns = df.select_dtypes(include="number").columns
categorical_columns = df.select_dtypes(include="str").columns

df[numeric_columns] = df[numeric_columns].fillna(
    df[numeric_columns].median()
)

for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("Missing values after handling:")
print(df.isnull().sum())

# 4. Train-Test Split
X = df.drop(["Will_Buy_EV", "Buyer_ID"], axis=1)
y = df["Will_Buy_EV"]

numeric_features = X.select_dtypes(include="number").columns
categorical_features = X.select_dtypes(include="str").columns

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric_features),
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Model Training
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC(kernel="rbf"))
])
model.fit(X_train, y_train)

# 6. Prediction
y_pred = model.predict(X_test)

# 7. Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))