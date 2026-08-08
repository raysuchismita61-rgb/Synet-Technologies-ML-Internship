import pandas as pd
df = pd.read_csv("titanic.csv")

print("========== ORIGINAL DATASET ==========")
print(df.head())

print("\nOriginal Shape:", df.shape)
print("\n========== MISSING VALUES BEFORE CLEANING ==========")
print(df.isnull().sum())
duplicates = df.duplicated().sum()

print("\nNumber of duplicate rows:", duplicates)

df = df.drop_duplicates()

print("Shape after removing duplicates:", df.shape)
# Fill missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked values with the mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Fill missing Cabin values with "Unknown"
df["Cabin"] = df["Cabin"].fillna("Unknown")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
df["Survived"] = df["Survived"].astype(int)
df["Pclass"] = df["Pclass"].astype(int)
df = df.rename(columns={
    "PassengerId": "Passenger_ID",
    "Survived": "Survived",
    "Pclass": "Passenger_Class",
    "Name": "Passenger_Name",
    "Sex": "Gender",
    "Age": "Age",
    "SibSp": "Siblings_Spouses",
    "Parch": "Parents_Children",
    "Ticket": "Ticket_Number",
    "Fare": "Fare",
    "Cabin": "Cabin_Number",
    "Embarked": "Port"
})

print("\n========== MISSING VALUES AFTER CLEANING ==========")
print(df.isnull().sum())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== CLEANED DATASET ==========")
print(df.head())

print("\nCleaned Dataset Shape:", df.shape)

df.to_csv("cleaned_titanic.csv", index=False)

print("\n========================================")
print("Data cleaning completed successfully!")
print("Cleaned dataset saved as: cleaned_titanic.csv")
print("========================================")