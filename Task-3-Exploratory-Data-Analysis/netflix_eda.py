import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("netflix_titles.csv")

print("========== NETFLIX DATASET ==========")

# Display first 5 rows
print(df.head())

# Dataset shape
print("\nDataset Shape:", df.shape)
print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe())

# Summary statistics for release year
print("\n========== RELEASE YEAR STATISTICS ==========")
print(df["release_year"].describe())

print("\n========== CONTENT TYPE COUNT ==========")
print(df["type"].value_counts())

# Bar chart for Movies vs TV Shows
type_counts = df["type"].value_counts()

plt.figure(figsize=(8, 5))

plt.bar(
    type_counts.index,
    type_counts.values
)

plt.title("Netflix Content Type Distribution")
plt.xlabel("Content Type")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig("content_type_distribution.png")
plt.show()
plt.close()

year_counts = df["release_year"].value_counts().sort_index()

plt.figure(figsize=(10, 5))

plt.plot(
    year_counts.index,
    year_counts.values
)

plt.title("Netflix Titles by Release Year")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig("release_year_trend.png")
plt.show()
plt.close()

recent_years = df[
    df["release_year"] >= 2010
]

recent_year_counts = (
    recent_years["release_year"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(10, 5))

plt.plot(
    recent_year_counts.index,
    recent_year_counts.values,
    marker="o"
)

plt.title("Netflix Titles Released Since 2010")
plt.xlabel("Release Year")
plt.ylabel("Number of Titles")

plt.tight_layout()
plt.savefig("recent_release_trend.png")
plt.show()
plt.close()

rating_counts = df["rating"].value_counts().head(10)

plt.figure(figsize=(10, 6))

plt.bar(
    rating_counts.index,
    rating_counts.values
)

plt.title("Top Netflix Content Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Titles")

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("rating_distribution.png")
plt.show()
plt.close()



df["release_year_numeric"] = pd.to_numeric(
    df["release_year"],
    errors="coerce"
)

# Extract numeric duration
df["duration_numeric"] = pd.to_numeric(
    df["duration"].str.extract(r"(\d+)")[0],
    errors="coerce"
)


correlation = df[
    [
        "release_year_numeric",
        "duration_numeric"
    ]
].corr()

print("\n========== CORRELATION MATRIX ==========")
print(correlation)
plt.figure(figsize=(7, 5))

plt.scatter(
    df["release_year_numeric"],
    df["duration_numeric"],
    alpha=0.3
)

plt.title("Release Year vs Duration")
plt.xlabel("Release Year")
plt.ylabel("Duration")

plt.tight_layout()
plt.savefig("correlation_scatter.png")
plt.show()
plt.close()
genre_counts = (
    df["listed_in"]
    .str.split(", ")
    .explode()
    .value_counts()
    .head(10)
)

print("\n========== TOP 10 GENRES ==========")
print(genre_counts)

plt.figure(figsize=(10, 6))

plt.barh(
    genre_counts.index[::-1],
    genre_counts.values[::-1]
)

plt.title("Top 10 Netflix Genres")
plt.xlabel("Number of Titles")
plt.ylabel("Genre")

plt.tight_layout()
plt.savefig("top_10_genres.png")
plt.show()
plt.close()

print("\n========== EDA INSIGHTS ==========")


most_common_type = df["type"].value_counts().idxmax()
most_common_type_count = df["type"].value_counts().max()

print(
    f"1. The most common content type is "
    f"{most_common_type} with {most_common_type_count} titles."
)


most_common_rating = df["rating"].value_counts().idxmax()

print(
    f"2. The most common content rating is "
    f"{most_common_rating}."
)


most_common_genre = genre_counts.idxmax()

print(
    f"3. The most common genre/category is "
    f"{most_common_genre}."
)


latest_year = df["release_year"].max()

print(
    f"4. The latest release year present in the dataset is "
    f"{latest_year}."
)


correlation_value = correlation.loc[
    "release_year_numeric",
    "duration_numeric"
]

print(
    f"5. The correlation between release year and duration is "
    f"{correlation_value:.3f}."
)

df.to_csv(
    "netflix_eda_processed.csv",
    index=False
)

print("\n========================================")
print("EDA completed successfully!")
print("Graphs saved successfully.")
print("Processed dataset saved as:")
print("netflix_eda_processed.csv")
print("========================================")