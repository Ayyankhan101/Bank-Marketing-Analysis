# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import io
import base64

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Marketing – Quick Insight Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# Load data (cached)
# --------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("bank-full.csv", sep=";")
    cat_cols = ['job','marital','education','default','housing','loan','contact','month','poutcome','y']
    for c in cat_cols:
        df[c] = df[c].astype('category')
    return df

df_bank = load_data()

# --------------------------------------------------
# Helper: KPI card
# --------------------------------------------------
def kpi_card(title, value, tip=None):
    st.markdown(
        f"""
        <div style="
            background:#f0f2f6;
            padding:10px 20px;
            border-radius:8px;
            margin-bottom:8px;
            border-left:5px solid #4F8BF9;">
            <h3 style='margin:0;font-size:20px;color:#333;'>{title}</h3>
            <p style='margin:0;font-size:28px;font-weight:bold;color:#000;'>{value}</p>
            <small style='color:#555;'>{tip or ''}</small>
        </div>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("🏦 Bank Marketing Dashboard")

# --------------------------------------------------
# Row 1 – KPIs
# --------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    kpi_card("Total Campaigns", f"{len(df_bank):,}")
with col2:
    kpi_card("Subscribed", f"{(df_bank['y'] == 'yes').sum():,}")
with col3:
    kpi_card("Conversion Rate", f"{(df_bank['y'] == 'yes').mean():.2%}")
with col4:
    kpi_card("Features", f"{df_bank.shape[1]-1}")

# --------------------------------------------------
# Row 2 – Class balance
# --------------------------------------------------
st.subheader("Class Balance (Target)")
fig, ax = plt.subplots(figsize=(3, 2))
sns.countplot(x='y', data=df_bank, palette="Set2", ax=ax)
ax.set_title("Subscription Outcome")
ax.set_xlabel("Subscribed")
ax.set_ylabel("Count")
st.pyplot(fig)

# --------------------------------------------------
# Row 3 – Numeric correlation heatmap
# --------------------------------------------------
st.subheader("Numeric Feature Correlations")
num_cols = df_bank.select_dtypes(include=[np.number]).columns
corr = df_bank[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    cmap='coolwarm',
    annot=True,
    fmt=".2f",
    annot_kws={"size": 7},
    vmin=-1, vmax=1,
    ax=ax
)
ax.set_title("Pearson Correlation")
st.pyplot(fig)

# --------------------------------------------------
# Row 4 – Conversion rates
# --------------------------------------------------
st.subheader("Conversion Rates by Key Categorical Features")

cat_vars = ['job', 'education', 'month', 'poutcome']

for i in range(0, len(cat_vars), 2):
    cols = st.columns(2)               # always 2 columns
    for j, c in enumerate(cat_vars[i:i+2]):
        with cols[j]:
            cr = (
                df_bank.groupby(c)['y']
                .apply(lambda s: (s == 'yes').mean())
                .sort_values(ascending=False)
                .head(6)
            )
            fig, ax = plt.subplots(figsize=(3, 2.5))
            sns.barplot(x=cr.values, y=cr.index, palette="viridis", ax=ax)
            ax.set_title(f"By {c.title()}")
            ax.set_xlabel("Conversion Rate")
            ax.set_ylabel("")
            st.pyplot(fig)

# --------------------------------------------------
# Row 5 – Quick baseline model & ROC
# --------------------------------------------------
st.subheader("Baseline Logistic Regression Performance")

@st.cache_data
def train_baseline():
    X = df_bank.drop(columns=['y'])
    y = (df_bank['y'] == 'yes').astype(int)

    cat_feats = X.select_dtypes(include=['category', 'object']).columns
    num_feats = X.select_dtypes(include=[np.number]).columns

    preprocess = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_feats),
            ('num', 'passthrough', num_feats)
        ]
    )

    model = Pipeline([
        ('prep', preprocess),
        ('clf', LogisticRegression(max_iter=1000))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    return auc, y_test, y_proba

auc, y_test, y_proba = train_baseline()

col_left, col_right = st.columns([1, 2])
with col_left:
    kpi_card("ROC-AUC", f"{auc:.3f}", "Higher is better (max 1.0)")

with col_right:
    fig, ax = plt.subplots(figsize=(4, 3))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    st.pyplot(fig)
