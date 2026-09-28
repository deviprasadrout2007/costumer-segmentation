import io
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

st.set_page_config(page_title="Retail Customer Segmentation", page_icon="🛍️", layout="wide")

REQUIRED = ["customerid", "invoiceno", "invoicedate", "quantity", "unitprice"]

st.title("🛍️ Retail Customer Segmentation")
st.caption("RFM analysis • K-Means clustering • Customer insights and marketing strategies")

@st.cache_data(show_spinner=False)
def load_and_clean(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = (df.columns.astype(str).str.strip().str.lower()
                  .str.replace(" ", "_", regex=False))
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df.dropna(subset=["customerid"]).copy()
    df["customerid"] = df["customerid"].astype(str).str.strip()
    df["invoiceno"] = df["invoiceno"].astype(str).str.strip()
    df["invoicedate"] = pd.to_datetime(df["invoicedate"], errors="coerce")
    for c in ["quantity", "unitprice"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["invoicedate", "quantity", "unitprice"])
    df = df.drop_duplicates()
    df = df[(df["unitprice"] > 0) & (df["quantity"] > 0)].copy()
    df["total_amount"] = df["quantity"] * df["unitprice"]
    df = df[np.isfinite(df["total_amount"]) & (df["total_amount"] > 0)].copy()
    if df.empty:
        raise ValueError("No valid positive-quantity transactions remain after cleaning.")
    return df

@st.cache_data(show_spinner=False)
def build_rfm(df):
    reference_date = df["invoicedate"].max() + pd.Timedelta(days=1)
    customers = (df.groupby("customerid").agg(
        Recency=("invoicedate", lambda x: (reference_date - x.max()).days),
        Frequency=("invoiceno", "nunique"),
        Monetary=("total_amount", "sum")
    ).reset_index())
    customers = customers[(customers.Recency >= 0) & (customers.Frequency > 0) & (customers.Monetary > 0)].copy()
    return customers

def marketing_strategy(segment):
    if "High Value" in segment:
        return "Loyalty rewards, personalized recommendations, exclusive offers and early access."
    if "Loyal" in segment:
        return "Loyalty points, repeat-purchase rewards, cross-selling and personalized recommendations."
    if "At Risk High-Value" in segment:
        return "Re-engagement campaigns, personalized offers and reminders based on past purchases."
    if "New or Occasional" in segment:
        return "Introductory offers, product recommendations and repeat-purchase incentives."
    if "Low Engagement" in segment:
        return "Targeted promotions and low-cost re-engagement campaigns."
    if "Frequent Low-Spend" in segment:
        return "Product bundles, cross-selling and volume-based offers."
    if "Inactive" in segment:
        return "Win-back campaigns, reminders and personalized offers."
    return "Regular promotions, personalized recommendations and loyalty incentives."

def classify_segment(row):
    r, f, m = row["Recency_Ratio"], row["Frequency_Ratio"], row["Monetary_Ratio"]
    if r <= .75 and f >= 1.20 and m >= 1.20:
        return "High Value Customers"
    elif r <= .85 and f >= 1.20:
        return "Loyal Customers"
    elif r >= 1.30 and m >= 1.20:
        return "At Risk High-Value Customers"
    elif r <= .85 and f < .80:
        return "New or Occasional Customers"
    elif r >= 1.30 and f < .80 and m < .80:
        return "Low Engagement Customers"
    elif f >= 1.20 and m < .80:
        return "Frequent Low-Spend Customers"
    elif r >= 1.50:
        return "Inactive Customers"
    return "Regular Customers"

@st.cache_data(show_spinner="Preparing customer segments...")
def train_model(customers, max_k):
    features = ["Recency", "Frequency", "Monetary"]
    rfm_log = np.log1p(customers[features].astype(float))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm_log)
    upper = min(max_k, len(customers) - 1)
    if upper < 2:
        raise ValueError("At least 3 customers are needed for clustering.")
    ks, inertias, scores = [], [], []
    for k in range(2, upper + 1):
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        ks.append(k)
        inertias.append(float(model.inertia_))
        # Silhouette is undefined if every row forms its own cluster.
        if len(np.unique(labels)) > 1 and len(np.unique(labels)) < len(labels):
            scores.append(float(silhouette_score(scaled, labels)))
        else:
            scores.append(-1.0)
    best_idx = int(np.argmax(scores))
    best_k = ks[best_idx]
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    result = customers.copy()
    result["Cluster"] = labels
    profile = result.groupby("Cluster").agg(
        Customers=("customerid", "count"),
        Avg_Recency=("Recency", "mean"),
        Avg_Frequency=("Frequency", "mean"),
        Avg_Monetary=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum")
    ).reset_index()
    overall = result[features].mean()
    profile["Recency_Ratio"] = profile["Avg_Recency"] / overall["Recency"]
    profile["Frequency_Ratio"] = profile["Avg_Frequency"] / overall["Frequency"]
    profile["Monetary_Ratio"] = profile["Avg_Monetary"] / overall["Monetary"]
    profile["Segment"] = profile.apply(classify_segment, axis=1)
    counts = profile["Segment"].value_counts()
    duplicates = counts[counts > 1].index
    for name in duplicates:
        mask = profile["Segment"] == name
        profile.loc[mask, "Segment"] = profile.loc[mask].apply(
            lambda row: f"{name} (Cluster {int(row['Cluster'])})", axis=1)
    profile["Marketing_Strategy"] = profile["Segment"].apply(marketing_strategy)
    mapping = dict(zip(profile["Cluster"], profile["Segment"]))
    result["Segment"] = result["Cluster"].map(mapping)
    profile = profile.round({"Avg_Recency": 2, "Avg_Frequency": 2, "Avg_Monetary": 2,
                             "Total_Revenue": 2, "Recency_Ratio": 3,
                             "Frequency_Ratio": 3, "Monetary_Ratio": 3})
    pca = PCA(n_components=2)
    coords = pca.fit_transform(scaled)
    pca_df = pd.DataFrame(coords, columns=["PCA1", "PCA2"])
    pca_df["Segment"] = result["Segment"].values
    return result, profile, ks, inertias, scores, best_k, scaler, model, mapping, pca_df

def predict_segment(r, f, m, scaler, model, mapping):
    vals = np.array([[r, f, m]], dtype=float)
    if np.any(vals < 0):
        raise ValueError("RFM inputs cannot be negative.")
    scaled = scaler.transform(np.log1p(vals))
    cluster = int(model.predict(scaled)[0])
    return mapping[cluster], cluster

uploaded = st.file_uploader("Upload retail transaction CSV", type=["csv"], help="Required columns: CustomerID, InvoiceNo, InvoiceDate, Quantity, UnitPrice")
with st.expander("Expected CSV format"):
    st.write("The column names are case-insensitive and spaces are converted to underscores.")
    st.code("CustomerID,InvoiceNo,InvoiceDate,Quantity,UnitPrice\n12346,536365,2010-12-01 08:26:00,6,2.55")

if uploaded is None:
    st.info("Upload your retail_data.csv file to begin. The app will clean the data, calculate RFM features, and train K-Means.")
    st.stop()

try:
    transactions = load_and_clean(uploaded.getvalue())
    customers = build_rfm(transactions)
except Exception as e:
    st.error(f"Could not process this dataset: {e}")
    st.stop()

if len(customers) < 3:
    st.error("At least 3 unique customers with valid transactions are required.")
    st.stop()

with st.sidebar:
    st.header("Model settings")
    max_k = st.slider("Maximum clusters to test", min_value=3, max_value=10, value=8)
    st.caption("The app tests K=2 through the selected maximum, limited by the number of customers.")
    run = st.button("Train / refresh model", type="primary", use_container_width=True)

# Cached model training runs when data or max_k changes. Button provides an explicit rerun control.
if "trained_key" not in st.session_state:
    st.session_state.trained_key = None
data_key = (uploaded.name, len(uploaded.getvalue()), max_k)
if run or st.session_state.trained_key != data_key:
    st.session_state.trained_key = data_key
    st.session_state.pop("model_result", None)
    try:
        st.session_state.model_result = train_model(customers, max_k)
    except Exception as e:
        st.error(f"Model training failed: {e}")
        st.stop()
if "model_result" not in st.session_state:
    st.session_state.model_result = train_model(customers, max_k)

result, profile, ks, inertias, scores, best_k, scaler, model, mapping, pca_df = st.session_state.model_result
total_revenue = transactions["total_amount"].sum()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Unique customers", f"{len(result):,}")
c2.metric("Valid transactions", f"{len(transactions):,}")
c3.metric("Total revenue", f"{total_revenue:,.2f}")
c4.metric("Selected clusters", str(best_k))
st.caption(f"Reference date for Recency: {(transactions.invoicedate.max() + pd.Timedelta(days=1)).date()}")

tabs = st.tabs(["Overview", "Exploratory Analysis", "Customer Segments", "Predict Customer", "Download Reports"])

with tabs[0]:
    st.subheader("Segment overview")
    st.dataframe(profile[["Cluster", "Segment", "Customers", "Avg_Recency", "Avg_Frequency", "Avg_Monetary", "Total_Revenue"]], use_container_width=True, hide_index=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    profile.set_index("Segment")["Customers"].plot(kind="bar", ax=ax)
    ax.set_ylabel("Customers")
    ax.set_xlabel("")
    ax.set_title("Customers in each segment")
    plt.xticks(rotation=30, ha="right")
    st.pyplot(fig, clear_figure=True)
    st.subheader("Marketing strategies")
    for _, row in profile.iterrows():
        with st.expander(str(row["Segment"])):
            st.write(row["Marketing_Strategy"])

with tabs[1]:
    st.subheader("Transaction analysis")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.histplot(transactions["total_amount"].clip(upper=transactions["total_amount"].quantile(.99)), bins=40, kde=True, ax=ax)
        ax.set_title("Transaction value (capped at 99th percentile)")
        ax.set_xlabel("Transaction value")
        st.pyplot(fig, clear_figure=True)
    with right:
        top = transactions.groupby("customerid")["total_amount"].sum().nlargest(10).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        top.plot(kind="barh", ax=ax)
        ax.set_title("Top 10 customers by revenue")
        ax.set_xlabel("Revenue")
        st.pyplot(fig, clear_figure=True)
    st.subheader("RFM distributions")
    for feature in ["Recency", "Frequency", "Monetary"]:
        fig, ax = plt.subplots(figsize=(8, 3))
        sns.histplot(result[feature], bins=35, kde=True, ax=ax)
        ax.set_title(f"{feature} distribution")
        st.pyplot(fig, clear_figure=True)
    st.subheader("Cluster selection diagnostics")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.plot(ks, inertias, marker="o")
        ax.set_title("Elbow method")
        ax.set_xlabel("K")
        ax.set_ylabel("Inertia")
        st.pyplot(fig, clear_figure=True)
    with right:
        fig, ax = plt.subplots()
        ax.plot(ks, scores, marker="o")
        ax.set_title("Silhouette scores")
        ax.set_xlabel("K")
        ax.set_ylabel("Score")
        st.pyplot(fig, clear_figure=True)
    st.success(f"Selected K={best_k}; silhouette score={scores[ks.index(best_k)]:.4f}")

with tabs[2]:
    st.subheader("Customer segment visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=result, x="Frequency", y="Monetary", hue="Segment", s=55, ax=ax)
    ax.set_title("Frequency vs Monetary value")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Segment", s=55, ax=ax)
    ax.set_title("PCA view of customer segments")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    st.pyplot(fig, clear_figure=True)
    selected = st.selectbox("Filter customers by segment", ["All"] + sorted(result["Segment"].unique().tolist()))
    view = result if selected == "All" else result[result["Segment"] == selected]
    st.dataframe(view, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Predict a new customer's segment")
    st.write("Enter RFM values using the same definitions as the uploaded dataset.")
    with st.form("prediction_form"):
        a, b, c = st.columns(3)
        recency = a.number_input("Recency (days)", min_value=0, value=20, step=1)
        frequency = b.number_input("Frequency (orders)", min_value=1, value=15, step=1)
        monetary = c.number_input("Monetary (total spend)", min_value=0.0, value=2500.0, step=100.0)
        submitted = st.form_submit_button("Predict segment", type="primary")
    if submitted:
        try:
            segment, cluster = predict_segment(recency, frequency, monetary, scaler, model, mapping)
            st.success(f"Predicted segment: {segment}")
            st.write(f"Assigned cluster: {cluster}")
            strategy = profile.loc[profile["Cluster"] == cluster, "Marketing_Strategy"].iloc[0]
            st.info("Suggested strategy: " + strategy)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

with tabs[4]:
    st.subheader("Download results")
    customer_csv = result.to_csv(index=False).encode("utf-8")
    profile_csv = profile[["Cluster", "Segment", "Customers", "Avg_Recency", "Avg_Frequency", "Avg_Monetary", "Total_Revenue", "Marketing_Strategy"]].to_csv(index=False).encode("utf-8")
    st.download_button("Download customer segments CSV", customer_csv, "customer_segments.csv", "text/csv")
    st.download_button("Download segment profile CSV", profile_csv, "segment_profile.csv", "text/csv")
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        result.to_excel(writer, sheet_name="Customer Segments", index=False)
        profile.to_excel(writer, sheet_name="Segment Profile", index=False)
    st.download_button("Download Excel report", excel_buffer.getvalue(), "customer_segmentation_report.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.caption("The uploaded file is processed during this session; generated reports are available through these download buttons.")

st.divider()
st.caption("Method: positive-quantity transactions only; RFM is calculated per customer; log1p transformation and StandardScaler precede K-Means. Segment labels are heuristic descriptions based on relative RFM averages.")
