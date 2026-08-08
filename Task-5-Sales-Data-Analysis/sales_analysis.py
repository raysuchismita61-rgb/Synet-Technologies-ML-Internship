import pandas as pd
import matplotlib.pyplot as plt

file_path = "Superstore.csv"

df = pd.read_csv(file_path)

print("\n========== SALES DATA ANALYSIS ==========\n")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())

print("\n========== DATASET INFORMATION ==========\n")

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

df["Order Date"] = pd.to_datetime(df["Order Date"])

print("\n========== MONTHLY REVENUE ==========\n")

monthly_revenue = (
    df.groupby(df["Order Date"].dt.to_period("M"))["Sales"]
    .sum()
)

print(monthly_revenue)
monthly_revenue.index = monthly_revenue.index.to_timestamp()


plt.figure(figsize=(12, 6))

plt.plot(
    monthly_revenue.index,
    monthly_revenue.values,
    marker="o"
)

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Revenue / Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

print("\n========== TOP-SELLING PRODUCTS ==========\n")

top_products = (
    df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products)


plt.figure(figsize=(12, 6))

plt.bar(
    top_products.index,
    top_products.values
)

plt.title("Top 10 Selling Products")
plt.xlabel("Product")
plt.ylabel("Total Sales")

plt.xticks(rotation=75)

plt.tight_layout()

plt.show()

print("\n========== PROFIT ANALYSIS ==========\n")

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()

print("Total Sales:", round(total_sales, 2))
print("Total Profit:", round(total_profit, 2))

profit_by_category = (
    df.groupby("Category")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nProfit by Category:")
print(profit_by_category)


# Plot profit by category
plt.figure(figsize=(8, 5))

plt.bar(
    profit_by_category.index,
    profit_by_category.values
)

plt.title("Profit by Category")
plt.xlabel("Category")
plt.ylabel("Profit")

plt.tight_layout()

plt.show()

profit_by_region = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)

print("\nProfit by Region:")
print(profit_by_region)

profit_margin = (
    total_profit / total_sales
) * 100

print(
    "\nOverall Profit Margin:",
    round(profit_margin, 2),
    "%"
)

print("\n========== BUSINESS INSIGHTS ==========\n")


best_month = monthly_revenue.idxmax()
best_month_value = monthly_revenue.max()

print(
    "1. Highest revenue month:",
    best_month.strftime("%B %Y"),
    "with sales of",
    round(best_month_value, 2)
)


best_product = top_products.idxmax()
best_product_sales = top_products.max()

print(
    "2. Top-selling product:",
    best_product,
    "with sales of",
    round(best_product_sales, 2)
)
best_category = profit_by_category.idxmax()
best_category_profit = profit_by_category.max()

print(
    "3. Most profitable category:",
    best_category,
    "with profit of",
    round(best_category_profit, 2)
)


best_region = profit_by_region.idxmax()
best_region_profit = profit_by_region.max()

print(
    "4. Most profitable region:",
    best_region,
    "with profit of",
    round(best_region_profit, 2)
)


print(
    "5. Overall profit margin:",
    round(profit_margin, 2),
    "%"
)


print("\n========== ANALYSIS COMPLETE ==========\n")