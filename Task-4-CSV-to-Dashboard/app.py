import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
st.set_page_config(
    page_title="Netflix Dashboard",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Netflix Dataset Dashboard")
st.write("Interactive dashboard for exploring Netflix titles.")


uploaded_file = st.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")

else:
    try:
        df = pd.read_csv("netflix_titles.csv")
        st.info("Using the default Netflix dataset.")
    except FileNotFoundError:
        st.warning("Please upload a CSV file to continue.")
        st.stop()

st.header("📊 Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Records", len(df))

with col2:
    st.metric("Total Columns", len(df.columns))

with col3:
    st.metric("Missing Values", int(df.isnull().sum().sum()))


# Show dataset
st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)

st.sidebar.header("🔎 Filters")


if "type" in df.columns:

    types = df["type"].dropna().unique().tolist()

    selected_type = st.sidebar.multiselect(
        "Select Type",
        types,
        default=types
    )

    filtered_df = df[df["type"].isin(selected_type)]

else:
    filtered_df = df.copy()



if "country" in filtered_df.columns:

    countries = (
        filtered_df["country"]
        .dropna()
        .str.split(", ")
        .explode()
        .unique()
    )

    selected_country = st.sidebar.selectbox(
        "Select Country",
        ["All"] + sorted(countries.tolist())
    )

    if selected_country != "All":
        filtered_df = filtered_df[
            filtered_df["country"]
            .fillna("")
            .str.contains(selected_country, case=False)
        ]


st.sidebar.write(
    f"Showing **{len(filtered_df)}** records"
)

st.header("📈 Interactive Charts")
if "type" in filtered_df.columns:

    st.subheader("Movies vs TV Shows")

    type_count = filtered_df["type"].value_counts()

    fig, ax = plt.subplots()

    ax.bar(
        type_count.index,
        type_count.values
    )

    ax.set_xlabel("Type")
    ax.set_ylabel("Number of Titles")
    ax.set_title("Movies vs TV Shows")

    st.pyplot(fig)
if "release_year" in filtered_df.columns:

    st.subheader("Titles by Release Year")

    year_count = (
        filtered_df["release_year"]
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots()

    ax.plot(
        year_count.index,
        year_count.values
    )

    ax.set_xlabel("Release Year")
    ax.set_ylabel("Number of Titles")
    ax.set_title("Netflix Titles by Release Year")

    st.pyplot(fig)
if "rating" in filtered_df.columns:

    st.subheader("Content Rating Distribution")

    rating_count = (
        filtered_df["rating"]
        .value_counts()
        .head(10)
    )

    fig, ax = plt.subplots()

    ax.bar(
        rating_count.index,
        rating_count.values
    )

    ax.set_xlabel("Rating")
    ax.set_ylabel("Number of Titles")
    ax.set_title("Top Content Ratings")

    plt.xticks(rotation=45)

    st.pyplot(fig)

st.header("📋 Summary Statistics")

st.write("Numerical columns:")

numeric_columns = filtered_df.select_dtypes(
    include="number"
).columns

if len(numeric_columns) > 0:
    st.dataframe(
        filtered_df[numeric_columns].describe(),
        use_container_width=True
    )
else:
    st.info("No numerical columns available.")
st.header("⚠️ Missing Values")

missing_values = (
    filtered_df.isnull()
    .sum()
    .sort_values(ascending=False)
)

missing_values = missing_values[missing_values > 0]

if len(missing_values) > 0:
    st.dataframe(
        missing_values.to_frame("Missing Values"),
        use_container_width=True
    )
else:
    st.success("No missing values found.")



st.markdown("---")
st.write("Created using Python, Pandas, Matplotlib and Streamlit.")