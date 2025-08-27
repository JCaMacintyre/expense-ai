import streamlit as st
import pandas as pd
import joblib
import io
from pathlib import Path
import re

# ---------- Config ----------
st.set_page_config(page_title="Expense AI", page_icon="💸", layout="wide")
MODEL_PATH = Path("models/expense_category_model.joblib")

# ---------- Helpers (mirror training-time cleaning) ----------
def clean_description(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.upper()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^A-Z0-9 &/.\-]", "", t)
    return t.strip()

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Try to map common names to our canonical ones
    lower_map = {c.lower().strip(): c for c in df.columns}
    date_col = lower_map.get("date") or lower_map.get("posted") or lower_map.get("transaction date")
    desc_col = lower_map.get("description") or lower_map.get("merchant") or lower_map.get("details")
    amt_col  = lower_map.get("amount") or lower_map.get("amt") or lower_map.get("value")

    rename = {}
    if date_col: rename[date_col] = "date"
    if desc_col: rename[desc_col] = "description"
    if amt_col:  rename[amt_col]  = "amount"
    if rename:
        df = df.rename(columns=rename)

    # Types
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if "amount" in df.columns:
        df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

    # Clean text
    if "description" in df.columns:
        df["description_clean"] = df["description"].map(clean_description)
    else:
        df["description"] = ""
        df["description_clean"] = ""

    # Keep useful columns up front
    cols = [c for c in ["date","description","amount","description_clean"] if c in df.columns]
    rest = [c for c in df.columns if c not in cols]
    df = df[cols + rest]
    return df

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

# ---------- UI ----------
st.title("💸 Expense AI — Smart Expense Analyzer")
st.caption("Upload a CSV of transactions → auto-categorize with your trained model → analyze and export.")

model = load_model()
if model is None:
    st.error("Model not found. Train it first by running `python day4_train.py` to create `models/expense_category_model.joblib`.")
    st.stop()

uploaded = st.file_uploader("Upload a CSV", type=["csv"])
if not uploaded:
    with st.expander("CSV format tips (works with bank exports or your labeled data)"):
        st.markdown(
            "- Include columns like **date**, **description**, and **amount** if you have them.\n"
            "- Amount: positive for income/credits, negative for expenses/debits.\n"
            "- If your download uses different headers, I’ll try to auto-detect them."
        )
    st.info("Choose a CSV to get started.")
    st.stop()

# Read CSV (handle odd encodings gracefully)
try:
    df_raw = pd.read_csv(uploaded)
except Exception:
    uploaded.seek(0)
    df_raw = pd.read_csv(uploaded, encoding_errors="ignore")

df = standardize_columns(df_raw.copy())

# Predict categories using the same field used to train
X = df["description_clean"].fillna("")
preds = model.predict(X)

df_out = df.copy()
df_out["predicted_category"] = preds

# ---------- Layout: Preview + Metrics ----------
st.subheader("Preview")
st.dataframe(df_out.head(25), use_container_width=True)

# Simple metrics
with st.container():
    col1, col2, col3 = st.columns(3)
    if "amount" in df_out:
        total_expenses = df_out.loc[df_out["amount"] < 0, "amount"].sum()
        total_income = df_out.loc[df_out["amount"] > 0, "amount"].sum()
        net = (total_income + total_expenses)
        col1.metric("Total Expenses", f"${abs(total_expenses):,.2f}")
        col2.metric("Total Income", f"${total_income:,.2f}")
        col3.metric("Net", f"${net:,.2f}")

# Category breakdown
st.subheader("Category Breakdown")
by_cat = df_out["predicted_category"].value_counts().rename_axis("category").reset_index(name="count")
st.dataframe(by_cat, use_container_width=True)

try:
    import plotly.express as px
    fig = px.bar(by_cat, x="category", y="count", title="Predicted Categories (count)")
    st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.caption("Install plotly for charts: `pip install plotly`")

# Optional: monthly trend if dates and amounts exist
if "date" in df_out and "amount" in df_out:
    st.subheader("Monthly Spend by Category (Expenses only)")
    dtemp = df_out.dropna(subset=["date"]).copy()
    if not dtemp.empty:
        dtemp["month"] = dtemp["date"].dt.to_period("M").astype(str)
        spend = dtemp.loc[dtemp["amount"] < 0].groupby(["month","predicted_category"])["amount"].sum().reset_index()
        spend["amount_abs"] = spend["amount"].abs()
        if not spend.empty:
            try:
                fig2 = px.bar(spend, x="month", y="amount_abs", color="predicted_category", title="Expenses by Month & Category", barmode="stack")
                st.plotly_chart(fig2, use_container_width=True)
            except Exception:
                pass

# ---------- Download ----------
st.subheader("Export")
out_csv = df_out.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV with Predictions",
    data=out_csv,
    file_name="transactions_categorized.csv",
    mime="text/csv"
)

st.success("Done! Upload a different CSV to re-run categorization.")
