import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor



# App settings

st.set_page_config(page_title="AQI Dashboard", layout="wide")

st.markdown("""
<style>
.block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
section[data-testid="stSidebar"] { padding-top: 1rem; }

[data-testid="stMetric"] {
    background: rgba(255,255,255,0.03);
    padding: 16px;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.06);
}

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

st.title("Air Quality Analysis & AQI Prediction App")
st.write(
    "This dashboard explores air quality measurements from multiple Indian cities "
    "and predicts AQI using a Random Forest model."
)



# Load + clean dataset

@st.cache_data
def load_data(csv_path="all_cities_merged.csv"):
    df = pd.read_csv(csv_path)

    # Date parsing
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # Create Year/Month
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month

    # Drop invalids + duplicates
    df = df.dropna(subset=["City", "Date"]).drop_duplicates()

    # Drop columns with >80% missing
    threshold = 0.80
    drop_cols = [c for c in df.columns if df[c].isna().mean() > threshold]
    df = df.drop(columns=drop_cols, errors="ignore")

    # City-wise median imputation for numeric columns
    numeric_cols = [
        c for c in df.columns
        if c not in ["City", "Date", "AQI_Bucket"]
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    for col in numeric_cols:
        df[col] = df.groupby("City")[col].transform(lambda x: x.fillna(x.median()))

    # Fix AQI bucket
    if "AQI_Bucket" in df.columns:
        df["AQI_Bucket"] = df["AQI_Bucket"].fillna("Unknown")

    return df


# Model training (cached)
# -------------------------------
@st.cache_resource(show_spinner=False)
def train_model(df):
    target = "AQI"

    candidate_features = [
        "PM2.5", "PM10", "NO", "NO2", "NOx", "NH3",
        "CO", "SO2", "O3", "Benzene", "Toluene",
        "City", "Year", "Month"
    ]
    features = [f for f in candidate_features if f in df.columns]

    df_model = df.dropna(subset=[target]).copy()
    X = df_model[features]
    y = df_model[target]

    cat_features = ["City"]
    num_features = [c for c in features if c != "City"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X, y)
    return model, features, num_features


def aqi_bucket(value):
    if value <= 50:
        return "Good"
    elif value <= 100:
        return "Satisfactory"
    elif value <= 200:
        return "Moderate"
    elif value <= 300:
        return "Poor"
    elif value <= 400:
        return "Very Poor"
    else:
        return "Severe"



# Load data + model

df = load_data()

with st.spinner("Training AQI prediction model... Please wait ⏳"):
    model, features, num_features = train_model(df)

cities = sorted(df["City"].dropna().unique())
years = sorted(df["Year"].dropna().astype(int).unique())



# Sidebar navigation

st.sidebar.title("Control Panel")
page = st.sidebar.radio("Go to", ["Dataset Overview", "City Explorer", "AQI Predictor"])
st.sidebar.divider()



# PAGE 1 — Dataset Overview

if page == "Dataset Overview":
    st.subheader("Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", f"{df.shape[1]:,}")
    c3.metric("Cities", f"{df['City'].nunique():,}")

    st.divider()

    with st.expander("View all cities in this dataset"):
        st.write(cities)

    st.subheader("Sample of Cleaned Dataset")
    st.caption("Pick a city to preview only that city’s cleaned records.")

    selected_city = st.selectbox("View sample data for:", ["All Cities"] + cities)
    sample_size = st.slider("Rows to display", 5, 30, 10)

    if selected_city == "All Cities":
        sample_df = df.sample(sample_size, random_state=42).sort_values("City")
    else:
        city_df = df[df["City"] == selected_city]
        sample_df = city_df.sample(min(sample_size, len(city_df)), random_state=42)

    st.dataframe(sample_df, use_container_width=True)

    st.divider()

    # AQI Bucket Donut/Bar
    if "AQI_Bucket" in df.columns:
        st.subheader("AQI Category Distribution")

        hide_unknown = st.checkbox("Hide 'Unknown' category", value=True)
        chart_type = st.radio("Chart style", ["Donut", "Bar"], horizontal=True)

        plot_df = df.copy()
        if hide_unknown:
            plot_df = plot_df[plot_df["AQI_Bucket"] != "Unknown"]

        bucket_counts = plot_df["AQI_Bucket"].value_counts().reset_index()
        bucket_counts.columns = ["AQI Category", "Count"]

        if chart_type == "Donut":
            fig = px.pie(
                bucket_counts,
                names="AQI Category",
                values="Count",
                hole=0.55
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")
            fig.update_layout(height=420, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.bar(
                bucket_counts.sort_values("Count", ascending=False),
                x="Count",
                y="AQI Category",
                orientation="h",
            )
            fig.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.caption("This view shows how frequently each AQI category appears in the dataset.")
    else:
        st.warning("AQI_Bucket column not found in this dataset.")



# PAGE 2 — City Explorer

elif page == "City Explorer":
    st.subheader("City Explorer (AQI Trends)")

    city = st.selectbox("Select a city:", cities)
    city_df = df[df["City"] == city].sort_values("Date")

    cA, cB, cC = st.columns(3)
    cA.metric("Records", f"{len(city_df):,}")
    cB.metric("Avg AQI", f"{city_df['AQI'].mean():.1f}")
    cC.metric("Max AQI", f"{city_df['AQI'].max():.0f}")

    st.divider()

    monthly = city_df.groupby(["Year", "Month"])["AQI"].mean().reset_index()
    monthly["YearMonth"] = pd.to_datetime(
        monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str) + "-01",
        errors="coerce"
    )

    fig = px.line(monthly, x="YearMonth", y="AQI", markers=True, title=f"Monthly Average AQI – {city}")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)



# PAGE 3 — AQI Predictor

elif page == "AQI Predictor":
    st.subheader("AQI Predictor")

    col1, col2, col3 = st.columns(3)
    with col1:
        city = st.selectbox("City", cities)
    with col2:
        year = st.selectbox("Year", years)
    with col3:
        month = st.selectbox("Month", list(range(1, 13)))

    st.write("Enter pollutant values below (defaults use dataset medians):")

    medians = df[num_features].median(numeric_only=True)

    inputs = {}
    left, right = st.columns(2)
    toggle = True

    for col in num_features:
        if col in ["Year", "Month"]:
            continue

        default_val = float(medians.get(col, 0.0))

        if toggle:
            with left:
                inputs[col] = st.number_input(col, value=default_val)
            toggle = False
        else:
            with right:
                inputs[col] = st.number_input(col, value=default_val)
            toggle = True

    if st.button("Predict AQI"):
        row = {}
        for f in features:
            if f == "City":
                row[f] = city
            elif f == "Year":
                row[f] = int(year)
            elif f == "Month":
                row[f] = int(month)
            else:
                row[f] = float(inputs.get(f, 0.0))

        X_input = pd.DataFrame([row])
        pred = float(model.predict(X_input)[0])

        c1, c2 = st.columns(2)
        c1.metric("Predicted AQI", f"{pred:.1f}")
        c2.metric("Predicted Category", aqi_bucket(pred))

