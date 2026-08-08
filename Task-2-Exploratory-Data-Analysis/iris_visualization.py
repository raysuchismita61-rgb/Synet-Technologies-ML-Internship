import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# --------------------------------------------------
# TASK 2: DATA VISUALIZATION
# Dataset: Iris Dataset
# --------------------------------------------------

# 1. Load Iris dataset
iris = load_iris()

# Create DataFrame
df = pd.DataFrame(
    iris.data,
    columns=[
        "Sepal_Length",
        "Sepal_Width",
        "Petal_Length",
        "Petal_Width"
    ]
)

# Add species names
df["Species"] = iris.target_names[iris.target]

# --------------------------------------------------
# 2. Display dataset information
# --------------------------------------------------

print("========== IRIS DATASET ==========")
print(df.head())

print("\nDataset Shape:", df.shape)

print("\nSpecies Count:")
print(df["Species"].value_counts())

# --------------------------------------------------
# 3. BAR CHART
# Average Petal Length by Species
# --------------------------------------------------

average_petal_length = df.groupby("Species")["Petal_Length"].mean()

plt.figure(figsize=(8, 5))
plt.bar(
    average_petal_length.index,
    average_petal_length.values
)

plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length (cm)")
plt.tight_layout()

plt.savefig("bar_chart.png")
plt.show()
plt.close()

# --------------------------------------------------
# 4. HISTOGRAM
# Distribution of Sepal Length
# --------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Sepal_Length"],
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig("histogram.png")
plt.show()
plt.close()

# --------------------------------------------------
# 5. SCATTER PLOT
# Sepal Length vs Petal Length
# --------------------------------------------------

plt.figure(figsize=(8, 5))

for species in df["Species"].unique():

    species_data = df[df["Species"] == species]

    plt.scatter(
        species_data["Sepal_Length"],
        species_data["Petal_Length"],
        label=species
    )

plt.title("Sepal Length vs Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()

plt.savefig("scatter_plot.png")
plt.show()
plt.close()

# --------------------------------------------------
# 6. FEATURE COMPARISON
# --------------------------------------------------

feature_comparison = df.groupby("Species")[
    [
        "Sepal_Length",
        "Sepal_Width",
        "Petal_Length",
        "Petal_Width"
    ]
].mean()

print("\n========== AVERAGE FEATURE VALUES ==========")
print(feature_comparison)

plt.figure(figsize=(10, 6))

feature_comparison.plot(kind="bar")

plt.title("Comparison of Average Iris Features by Species")
plt.xlabel("Species")
plt.ylabel("Average Measurement (cm)")
plt.xticks(rotation=0)
plt.legend(title="Features")
plt.tight_layout()

plt.savefig("feature_comparison.png")
plt.show()
plt.close()

# --------------------------------------------------
# 7. Save Iris dataset
# --------------------------------------------------

df.to_csv("iris_dataset.csv", index=False)

# --------------------------------------------------
# 8. Final message
# --------------------------------------------------

print("\n========================================")
print("Data visualization completed successfully!")
print("Charts saved successfully.")
print("Iris dataset saved as: iris_dataset.csv")
print("========================================")