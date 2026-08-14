import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

DATA_PATH = "data/customer_churn.csv"

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("KAGGLE CUSTOMER CHURN ANALYSIS")
print("=" * 60)

print("\nDataset shape:")
print(df.shape)

print("\nFirst five records:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())


# --------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------

# Convert TotalCharges to numeric if it exists
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

# Fill missing numeric values with median
numeric_columns = df.select_dtypes(include=["number"]).columns

for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values with mode
categorical_columns = df.select_dtypes(include=["object"]).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].mode()[0])


# --------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS
# --------------------------------------------------

print("\nSummary statistics:")
print(df.describe())

if "Churn" in df.columns:
    print("\nChurn distribution:")
    print(df["Churn"].value_counts())

    print("\nChurn percentage:")
    print(df["Churn"].value_counts(normalize=True) * 100)


# --------------------------------------------------
# 4. VISUALIZATION
# --------------------------------------------------

os.makedirs("images", exist_ok=True)

if "Churn" in df.columns:

    plt.figure(figsize=(7, 5))

    df["Churn"].value_counts().plot(
        kind="bar"
    )

    plt.title("Customer Churn Distribution")
    plt.xlabel("Churn")
    plt.ylabel("Number of Customers")
    plt.tight_layout()

    plt.savefig("images/churn_distribution.png")

    plt.close()

    print(
        "\nSaved visualization:"
        " images/churn_distribution.png"
    )


# --------------------------------------------------
# 5. PREPARE DATA FOR MACHINE LEARNING
# --------------------------------------------------

model_df = df.copy()

# Remove customer identifier
if "customerID" in model_df.columns:
    model_df = model_df.drop(columns=["customerID"])

# Encode categorical variables
encoder = LabelEncoder()

for column in model_df.select_dtypes(include=["object"]).columns:
    model_df[column] = encoder.fit_transform(
        model_df[column].astype(str)
    )


# --------------------------------------------------
# 6. TRAIN MODEL
# --------------------------------------------------

if "Churn" in model_df.columns:

    X = model_df.drop(columns=["Churn"])
    y = model_df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

    print(f"\nAccuracy: {accuracy:.2%}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


    # --------------------------------------------------
    # 7. FEATURE IMPORTANCE
    # --------------------------------------------------

    importance = pd.Series(
        model.feature_importances_,
        index=X.columns
    ).sort_values(
        ascending=False
    ).head(10)

    print("\nTop 10 Predictive Features:")
    print(importance)

    plt.figure(figsize=(9, 6))

    importance.sort_values().plot(
        kind="barh"
    )

    plt.title(
        "Top 10 Features Affecting Customer Churn"
    )

    plt.xlabel("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        "images/feature_importance.png"
    )

    plt.close()

    print(
        "\nSaved visualization:"
        " images/feature_importance.png"
    )


print("\nAnalysis complete.")