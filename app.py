import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib, json, os, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (roc_curve, precision_recall_curve, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils import load_data, load_metrics, load_champion, load_scaler, load_feature_columns, preprocess_input

BASE = os.path.dirname(os.path.abspath(__file__))

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Intelligence Platform",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0a0a1a; }
    .stApp { background: linear-gradient(135deg, #0a0a1a 0%, #0f0f2e 100%); }

    .metric-card {
        background: linear-gradient(135deg, #1a1a3e, #0f0f2e);
        border: 1px solid #3a3a6a;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(123,47,255,0.15);
    }
    .metric-card h3 { color: #a0a0ff; font-size: 0.85rem; font-weight: 600; margin: 0 0 0.3rem 0; letter-spacing: 1px; text-transform: uppercase; }
    .metric-card .value { color: #ffffff; font-size: 2rem; font-weight: 700; }
    .metric-card .delta { color: #00d4ff; font-size: 0.8rem; margin-top: 0.2rem; }

    .hero-title { font-size: 2.8rem; font-weight: 700; background: linear-gradient(90deg, #7b2fff, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .hero-sub { color: #8888bb; font-size: 1.1rem; }

    .section-header { color: #7b2fff; font-size: 1.4rem; font-weight: 700; border-left: 4px solid #7b2fff; padding-left: 0.8rem; margin: 1.5rem 0 1rem 0; }
    .champion-badge { background: linear-gradient(135deg,#7b2fff,#00d4ff); color:white; padding:0.3rem 0.8rem; border-radius:20px; font-size:0.8rem; font-weight:700; }
    .churn-high { background:#ff4757; color:white; padding:0.8rem 1.5rem; border-radius:10px; font-size:1.2rem; font-weight:700; text-align:center; }
    .churn-low  { background:#2ed573; color:#0a0a1a; padding:0.8rem 1.5rem; border-radius:10px; font-size:1.2rem; font-weight:700; text-align:center; }

    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0f0f2e, #1a0a3e) !important; border-right: 1px solid #3a3a6a; }
    [data-testid="stSidebar"] .css-1d391kg { padding: 1rem; }

    div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label { color: #a0a0ff !important; font-weight: 600 !important; }
    .stButton > button { background: linear-gradient(135deg, #7b2fff, #00d4ff); color: white; border: none; border-radius: 8px; font-weight: 700; padding: 0.6rem 2rem; font-size: 1rem; cursor: pointer; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(123,47,255,0.4); }

    .stDataFrame { border: 1px solid #3a3a6a !important; border-radius: 8px; }
    h1, h2, h3 { color: #e0e0ff !important; }
    p, li, label { color: #c0c0dd !important; }
    .stMarkdown p { color: #c0c0dd !important; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Navigation")
    page = st.radio("", [
        "🏠  Home",
        "📊  Dataset Explorer",
        "🔍  EDA Dashboard",
        "🤖  Model Training",
        "🏆  Model Comparison",
        "🎯  Prediction System"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### 📌 Project Info")
    st.markdown("**Dataset:** Telco Customer Churn")
    st.markdown("**Records:** 7,032")
    st.markdown("**Features:** 20")
    st.markdown("**Models:** 6 + 2 Tuned")
    st.markdown("---")
    st.markdown("**Champion Model:** XGBoost Tuned")
    st.markdown("**AUC Score:** 0.838")

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
@st.cache_data
def get_data():
    return load_data()

@st.cache_resource
def get_champion():
    return load_champion()

df = get_data()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown('<div class="hero-title">📡 Telco Churn Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">End-to-End Customer Churn Prediction & Comparative ML Analysis</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    churn_rate = (df['Churn'] == 'Yes').mean()
    with col1:
        st.markdown(f'<div class="metric-card"><h3>Total Customers</h3><div class="value">7,032</div><div class="delta">Cleaned dataset</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>Churn Rate</h3><div class="value">{churn_rate:.1%}</div><div class="delta">26.6% at risk</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>Champion Model</h3><div class="value">XGBoost</div><div class="delta">AUC = 0.838</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><h3>Features</h3><div class="value">20</div><div class="delta">After engineering</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([3, 2])

    with col_a:
        st.markdown('<div class="section-header">🎯 Business Problem</div>', unsafe_allow_html=True)
        st.markdown("""
        Customer churn — the rate at which customers stop doing business with a company — is a critical
        KPI for telecom providers. **Acquiring a new customer costs 5-7× more** than retaining an existing one.

        This platform delivers a **production-grade churn prediction system** that identifies high-risk
        customers before they leave, enabling targeted retention interventions.

        **Key Business Questions:**
        - Which customers are most likely to churn in the next billing cycle?
        - What service or contract factors drive churn behavior?
        - How accurately can ML models distinguish churners from retained accounts?
        """)

        st.markdown('<div class="section-header">🗺️ Project Pipeline</div>', unsafe_allow_html=True)
        phases = [
            ("Phase 1-2", "Problem Understanding & Dataset Profiling"),
            ("Phase 3",   "Exploratory Data Analysis (EDA)"),
            ("Phase 4",   "Data Preprocessing & Feature Engineering"),
            ("Phase 5",   "6-Model ML Training & Evaluation"),
            ("Phase 6",   "Comparative Analysis & Champion Selection"),
            ("Phase 7",   "GridSearchCV Hyperparameter Tuning"),
            ("Phase 8",   "Streamlit Interactive Dashboard"),
            ("Phase 9",   "Cloud Deployment"),
        ]
        for phase, desc in phases:
            st.markdown(f"✅ **{phase}** — {desc}")

    with col_b:
        st.markdown('<div class="section-header">📋 Dataset Overview</div>', unsafe_allow_html=True)
        feature_groups = {
            "👤 Demographics": ["gender", "SeniorCitizen", "Partner", "Dependents"],
            "📱 Services": ["PhoneService", "MultipleLines", "InternetService"],
            "🔒 Add-ons": ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"],
            "🎬 Entertainment": ["StreamingTV", "StreamingMovies"],
            "📄 Account": ["Contract", "PaperlessBilling", "PaymentMethod"],
            "💰 Charges": ["tenure", "MonthlyCharges", "TotalCharges"],
        }
        for group, features in feature_groups.items():
            with st.expander(group):
                for f in features:
                    st.markdown(f"• `{f}`")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Dataset Explorer":
    st.markdown('<div class="hero-title">📊 Dataset Explorer</div>', unsafe_allow_html=True)
    st.markdown("Interactive data profiling — filter, inspect, and understand the raw dataset.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data View", "📈 Statistics", "🔬 Column Profiles", "⚠️ Data Quality"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        with col1:
            churn_filter = st.selectbox("Filter by Churn", ["All", "Yes", "No"])
        with col2:
            contract_filter = st.selectbox("Filter by Contract", ["All"] + list(df['Contract'].unique()))
        with col3:
            n_rows = st.slider("Rows to display", 10, 200, 50)

        filtered = df.copy()
        if churn_filter != "All":
            filtered = filtered[filtered['Churn'] == churn_filter]
        if contract_filter != "All":
            filtered = filtered[filtered['Contract'] == contract_filter]

        st.markdown(f"**Showing {len(filtered):,} records** | Shape: `{filtered.shape}`")
        st.dataframe(filtered.head(n_rows), use_container_width=True, height=420)

    with tab2:
        st.markdown("### Statistical Summary")
        df_numeric = df.select_dtypes(include=[np.number])
        st.dataframe(df_numeric.describe().T.style.format("{:.3f}"), use_container_width=True)

        st.markdown("### Categorical Value Counts")
        cat_col = st.selectbox("Select column", df.select_dtypes('object').columns.tolist())
        vc = df[cat_col].value_counts().reset_index()
        vc.columns = [cat_col, 'Count']
        vc['Percentage'] = (vc['Count'] / len(df) * 100).round(2)
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(vc, use_container_width=True)
        with col2:
            fig = px.pie(vc, names=cat_col, values='Count',
                         color_discrete_sequence=px.colors.sequential.Plasma,
                         title=f"{cat_col} Distribution")
            fig.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e',
                              font_color='white', title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("### Column Profile")
        for col in df.columns:
            with st.expander(f"📌 `{col}` — {df[col].dtype}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Nulls", df[col].isnull().sum())
                c2.metric("Unique Values", df[col].nunique())
                c3.metric("Type", str(df[col].dtype))
                if df[col].dtype in [np.float64, np.int64]:
                    c4.metric("Mean", f"{df[col].mean():.2f}")
                else:
                    c4.metric("Top Value", df[col].mode()[0])

    with tab4:
        st.markdown("### Data Quality Report")
        quality_data = {
            'Column': df.columns.tolist(),
            'Dtype': [str(df[c].dtype) for c in df.columns],
            'Nulls': [df[c].isnull().sum() for c in df.columns],
            'Null%': [(df[c].isnull().sum()/len(df)*100).round(2) for c in df.columns],
            'Unique': [df[c].nunique() for c in df.columns],
        }
        quality_df = pd.DataFrame(quality_data)
        st.dataframe(quality_df.style.highlight_between(subset='Nulls', left=1, color='#ff4444'),
                     use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", f"{len(df):,}")
        col2.metric("Total Columns", len(df.columns))
        col3.metric("Duplicate Rows", df.duplicated().sum())

        st.info("✅ TotalCharges: 11 rows had whitespace values (tenure=0). These were dropped during preprocessing.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍  EDA Dashboard":
    st.markdown('<div class="hero-title">🔍 EDA Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Interactive exploratory analysis with live charts.")
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Univariate", "🔗 Bivariate", "🌡️ Correlation", "🎯 Target Analysis"])

    with tab1:
        st.markdown("### Distribution Analysis")
        col1, col2 = st.columns([1, 3])
        with col1:
            num_col = st.selectbox("Select Feature", ['tenure','MonthlyCharges','TotalCharges'])
            chart_type = st.radio("Chart Type", ["Histogram","Box Plot","Violin"])
            split_churn = st.checkbox("Split by Churn", value=True)
        with col2:
            if chart_type == "Histogram":
                fig = px.histogram(df, x=num_col, color='Churn' if split_churn else None,
                                   nbins=40, barmode='overlay', opacity=0.75,
                                   color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                                   title=f"{num_col} Distribution")
            elif chart_type == "Box Plot":
                fig = px.box(df, x='Churn' if split_churn else None, y=num_col,
                             color='Churn' if split_churn else None,
                             color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                             title=f"{num_col} Box Plot")
            else:
                fig = px.violin(df, x='Churn' if split_churn else None, y=num_col,
                                color='Churn' if split_churn else None, box=True,
                                color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                                title=f"{num_col} Violin Plot")
            fig.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e',
                              font_color='white', title_font_color='white')
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Senior Citizen Distribution")
        df_sc = df.copy()
        df_sc['SeniorCitizen'] = df_sc['SeniorCitizen'].map({0:'Non-Senior',1:'Senior'})
        fig2 = px.bar(df_sc.groupby(['SeniorCitizen','Churn']).size().reset_index(name='Count'),
                      x='SeniorCitizen', y='Count', color='Churn', barmode='group',
                      color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                      title="Senior Citizen vs Churn")
        fig2.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("### Bivariate Analysis — Categorical Feature vs Churn")
        cat_features = ['Contract','PaymentMethod','InternetService','gender','Partner',
                        'Dependents','PhoneService','PaperlessBilling','MultipleLines',
                        'OnlineSecurity','TechSupport','StreamingTV']
        selected = st.selectbox("Select Feature", cat_features)
        ct = pd.crosstab(df[selected], df['Churn']).reset_index()
        ct_melt = ct.melt(id_vars=selected, var_name='Churn', value_name='Count')
        fig = px.bar(ct_melt, x=selected, y='Count', color='Churn', barmode='group',
                     color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                     title=f"{selected} vs Churn")
        fig.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e',
                          font_color='white', xaxis_tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Churn Rate by Category (%)")
        rate_df = df.groupby(selected)['Churn'].apply(lambda x: (x=='Yes').mean()*100).reset_index()
        rate_df.columns = [selected, 'Churn Rate (%)']
        rate_df = rate_df.sort_values('Churn Rate (%)', ascending=False)
        fig2 = px.bar(rate_df, x=selected, y='Churn Rate (%)',
                      color='Churn Rate (%)', color_continuous_scale='RdYlGn_r',
                      title=f"Churn Rate % by {selected}")
        fig2.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### Correlation Matrix")
        df_corr = df.copy()
        df_corr['TotalCharges'] = pd.to_numeric(df_corr['TotalCharges'], errors='coerce')
        binary_map_corr = {'Yes':1,'No':0,'Male':1,'Female':0,'No phone service':0,'No internet service':0}
        for col in df_corr.select_dtypes('object').columns:
            if col not in ['customerID']:
                df_corr[col] = df_corr[col].map(lambda x: binary_map_corr.get(x,x))
                df_corr[col] = pd.factorize(df_corr[col])[0]
        df_corr = df_corr.drop(columns=['customerID'], errors='ignore')
        corr = df_corr.corr()

        fig = px.imshow(corr, text_auto='.2f', aspect='auto',
                        color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        title="Feature Correlation Heatmap")
        fig.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e',
                          font_color='white', title_font_size=16, height=700)
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top Correlations with Churn")
        churn_corr = corr['Churn'].drop('Churn').sort_values(key=abs, ascending=False).head(10)
        fig2 = px.bar(x=churn_corr.index, y=churn_corr.values,
                      color=churn_corr.values, color_continuous_scale='RdBu_r',
                      labels={'x':'Feature','y':'Correlation with Churn'},
                      title="Top 10 Features Correlated with Churn")
        fig2.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e', font_color='white')
        st.plotly_chart(fig2, use_container_width=True)

    with tab4:
        st.markdown("### Target Class Distribution")
        churn_counts = df['Churn'].value_counts().reset_index()
        churn_counts.columns = ['Churn', 'Count']
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(churn_counts, names='Churn', values='Count',
                         color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                         title="Churn Distribution", hole=0.4)
            fig.update_layout(paper_bgcolor='#0f0f1a', font_color='white')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(churn_counts, x='Churn', y='Count',
                          color='Churn', color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                          title="Churn Count", text='Count')
            fig2.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### Scatter: Tenure vs Monthly Charges")
        fig3 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                          color_discrete_map={'Yes':'#ff6b6b','No':'#00d4ff'},
                          opacity=0.5, title="Tenure vs Monthly Charges by Churn",
                          marginal_x='histogram', marginal_y='histogram')
        fig3.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e', font_color='white')
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Model Training":
    st.markdown('<div class="hero-title">🤖 Model Training</div>', unsafe_allow_html=True)
    st.markdown("Select a model, configure parameters, and train on-the-fly.")
    st.markdown("---")

    @st.cache_data
    def get_preprocessed():
        df_raw = load_data()
        df_raw.drop(columns=['customerID'], inplace=True)
        df_raw['Churn'] = (df_raw['Churn'] == 'Yes').astype(int)
        binary_cols = ['Partner','Dependents','PhoneService','PaperlessBilling',
                       'MultipleLines','OnlineSecurity','OnlineBackup','DeviceProtection',
                       'TechSupport','StreamingTV','StreamingMovies']
        for col in binary_cols:
            df_raw[col] = df_raw[col].map({'Yes':1,'No':0,'No phone service':0,'No internet service':0})
        df_raw['gender'] = df_raw['gender'].map({'Male':1,'Female':0})
        df_raw = pd.get_dummies(df_raw, columns=['InternetService','Contract','PaymentMethod'])
        scaler = load_scaler()
        df_raw[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(df_raw[['tenure','MonthlyCharges','TotalCharges']])
        X = df_raw.drop('Churn', axis=1)
        y = df_raw['Churn']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train, X_test, y_train, y_test = get_preprocessed()

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### ⚙️ Model Configuration")
        model_choice = st.selectbox("Select Model", [
            "Logistic Regression", "Decision Tree", "Random Forest",
            "K-Nearest Neighbors", "SVM", "XGBoost"
        ])

        params = {}
        if model_choice == "Logistic Regression":
            params['C'] = st.slider("Regularization (C)", 0.01, 10.0, 1.0)
            params['max_iter'] = st.slider("Max Iterations", 100, 2000, 1000)
        elif model_choice == "Decision Tree":
            params['max_depth'] = st.slider("Max Depth", 2, 20, 8)
            params['min_samples_split'] = st.slider("Min Samples Split", 2, 20, 2)
        elif model_choice == "Random Forest":
            params['n_estimators'] = st.slider("N Estimators", 50, 300, 100)
            params['max_depth'] = st.slider("Max Depth", 2, 20, 10)
        elif model_choice == "K-Nearest Neighbors":
            params['n_neighbors'] = st.slider("K Neighbors", 1, 25, 7)
        elif model_choice == "SVM":
            params['C'] = st.slider("Regularization (C)", 0.1, 10.0, 1.0)
        elif model_choice == "XGBoost":
            params['n_estimators'] = st.slider("N Estimators", 50, 300, 100)
            params['max_depth'] = st.slider("Max Depth", 2, 10, 4)
            params['learning_rate'] = st.slider("Learning Rate", 0.01, 0.3, 0.05)

        train_btn = st.button("🚀 Train Model")

    with col2:
        if train_btn:
            with st.spinner(f"Training {model_choice}..."):
                if model_choice == "Logistic Regression":
                    model = LogisticRegression(**params, random_state=42)
                elif model_choice == "Decision Tree":
                    model = DecisionTreeClassifier(**params, random_state=42)
                elif model_choice == "Random Forest":
                    model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                elif model_choice == "K-Nearest Neighbors":
                    model = KNeighborsClassifier(**params)
                elif model_choice == "SVM":
                    model = SVC(**params, probability=True, random_state=42)
                elif model_choice == "XGBoost":
                    model = XGBClassifier(**params, random_state=42, use_label_encoder=False, eval_metric='logloss', verbosity=0)

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:,1]

                acc   = accuracy_score(y_test, y_pred)
                prec  = precision_score(y_test, y_pred)
                rec   = recall_score(y_test, y_pred)
                f1    = f1_score(y_test, y_pred)
                auc   = roc_auc_score(y_test, y_prob)

            st.success(f"✅ {model_choice} trained successfully!")
            m1,m2,m3,m4,m5 = st.columns(5)
            m1.metric("Accuracy",  f"{acc:.3f}")
            m2.metric("Precision", f"{prec:.3f}")
            m3.metric("Recall",    f"{rec:.3f}")
            m4.metric("F1 Score",  f"{f1:.3f}")
            m5.metric("ROC-AUC",   f"{auc:.3f}")

            c1, c2 = st.columns(2)
            with c1:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5,4))
                fig.patch.set_facecolor('#1a1a2e')
                ax.set_facecolor('#1a1a2e')
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                            xticklabels=['No Churn','Churn'],
                            yticklabels=['No Churn','Churn'],
                            annot_kws={'size':14,'weight':'bold'},
                            linewidths=1, linecolor='#0f0f1a')
                ax.set_title('Confusion Matrix', color='white', fontweight='bold')
                ax.tick_params(colors='white'); ax.xaxis.label.set_color('white'); ax.yaxis.label.set_color('white')
                st.pyplot(fig, use_container_width=True)
                plt.close()

            with c2:
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.3f})',
                                             line=dict(color='#7b2fff', width=3)))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines',
                                             line=dict(color='gray', dash='dash'), name='Baseline'))
                fig_roc.update_layout(title='ROC Curve', paper_bgcolor='#1a1a2e',
                                      plot_bgcolor='#1a1a2e', font_color='white',
                                      xaxis_title='FPR', yaxis_title='TPR', height=350)
                st.plotly_chart(fig_roc, use_container_width=True)
        else:
            st.info("👈 Configure model parameters and click **Train Model** to see results.")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆  Model Comparison":
    st.markdown('<div class="hero-title">🏆 Model Comparison Dashboard</div>', unsafe_allow_html=True)
    st.markdown("Full comparative analysis across all 8 trained models.")
    st.markdown("---")

    metrics_raw = load_metrics()
    model_names = list(metrics_raw.keys())
    metric_keys = ['Accuracy','Precision','Recall','F1','ROC-AUC']
    results_df = pd.DataFrame(metrics_raw).T.reset_index().rename(columns={'index':'Model'})

    # Champion callout
    champion = results_df.loc[results_df['ROC-AUC'].idxmax()]
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#1a1a3e,#0a0a2a);border:2px solid #7b2fff;border-radius:12px;padding:1rem 1.5rem;margin-bottom:1.5rem;">
        <span style="color:#7b2fff;font-size:1.2rem;font-weight:700;">🏆 Champion Model: {champion['Model']}</span>
        &nbsp;&nbsp;<span class="champion-badge">PRODUCTION READY</span><br>
        <span style="color:#aaa;font-size:0.9rem;">
        Accuracy: {champion['Accuracy']:.3f} &nbsp;|&nbsp;
        F1: {champion['F1']:.3f} &nbsp;|&nbsp;
        ROC-AUC: {champion['ROC-AUC']:.3f} &nbsp;|&nbsp;
        Recall: {champion['Recall']:.3f}
        </span>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Metrics Table", "📈 Bar Charts", "📉 ROC Curves", "🧩 Confusion Matrices", "🎯 PR Curves"])

    with tab1:
        st.markdown("### Performance Leaderboard")
        display_df = results_df.copy()
        for c in metric_keys:
            display_df[c] = display_df[c].apply(lambda x: f"{float(x):.4f}")
        st.dataframe(display_df.set_index('Model'), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(results_df.melt(id_vars='Model', value_vars=['Accuracy','F1'],
                                         var_name='Metric', value_name='Score'),
                         x='Model', y='Score', color='Metric', barmode='group',
                         color_discrete_map={'Accuracy':'#00d4ff','F1':'#ff6b6b'},
                         title="Accuracy vs F1 Score")
            fig.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e',
                              font_color='white', xaxis_tickangle=-30)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(results_df.sort_values('ROC-AUC', ascending=True),
                          x='ROC-AUC', y='Model', orientation='h',
                          color='ROC-AUC', color_continuous_scale='Viridis',
                          title="Models Ranked by ROC-AUC")
            fig2.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e', font_color='white')
            st.plotly_chart(fig2, use_container_width=True)

        all_metrics_long = results_df.melt(id_vars='Model', value_vars=metric_keys,
                                            var_name='Metric', value_name='Score')
        fig3 = px.bar(all_metrics_long, x='Model', y='Score', color='Metric', barmode='group',
                      color_discrete_sequence=['#7b2fff','#ff6b6b','#00d4ff','#ffd700','#00ff9f'],
                      title="All 5 Metrics Comparison")
        fig3.update_layout(paper_bgcolor='#0f0f1a', plot_bgcolor='#1a1a2e',
                           font_color='white', xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    with tab3:
        img_path = os.path.join(BASE, 'visuals', '07_roc_curves.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning("ROC curve image not found. Please run preprocessing script first.")

    with tab4:
        img_path = os.path.join(BASE, 'visuals', '08_confusion_matrices.png')
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning("Confusion matrix image not found.")

    with tab5:
        col_a, col_b = st.columns(2)
        with col_a:
            img_path = os.path.join(BASE, 'visuals', '09_precision_recall_curves.png')
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)
        with col_b:
            img_path = os.path.join(BASE, 'visuals', '10_feature_importance.png')
            if os.path.exists(img_path):
                st.image(img_path, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — PREDICTION SYSTEM
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Prediction System":
    st.markdown('<div class="hero-title">🎯 Churn Prediction System</div>', unsafe_allow_html=True)
    st.markdown("Enter customer details below to get a real-time churn risk prediction.")
    st.markdown("---")

    col_form, col_result = st.columns([3, 2])

    with col_form:
        st.markdown("### 👤 Customer Demographics")
        c1, c2, c3 = st.columns(3)
        with c1:
            gender     = st.selectbox("Gender", ["Male","Female"])
            senior     = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with c2:
            partner    = st.selectbox("Has Partner", ["Yes","No"])
            dependents = st.selectbox("Has Dependents", ["Yes","No"])
        with c3:
            tenure = st.slider("Tenure (Months)", 0, 72, 12)

        st.markdown("### 📱 Services Subscribed")
        c4, c5, c6 = st.columns(3)
        with c4:
            phone_service   = st.selectbox("Phone Service", ["Yes","No"])
            multiple_lines  = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
            internet        = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        with c5:
            online_security = st.selectbox("Online Security", ["Yes","No","No internet service"])
            online_backup   = st.selectbox("Online Backup",   ["Yes","No","No internet service"])
            device_protect  = st.selectbox("Device Protection",["Yes","No","No internet service"])
        with c6:
            tech_support    = st.selectbox("Tech Support",    ["Yes","No","No internet service"])
            streaming_tv    = st.selectbox("Streaming TV",    ["Yes","No","No internet service"])
            streaming_movies= st.selectbox("Streaming Movies",["Yes","No","No internet service"])

        st.markdown("### 💳 Account & Billing")
        c7, c8, c9 = st.columns(3)
        with c7:
            contract   = st.selectbox("Contract Type", ["Month-to-month","One year","Two year"])
            paperless  = st.selectbox("Paperless Billing", ["Yes","No"])
        with c8:
            payment    = st.selectbox("Payment Method", ["Electronic check","Mailed check",
                                                          "Bank transfer (automatic)","Credit card (automatic)"])
        with c9:
            monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
            total_charges   = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                               value=float(round(tenure * monthly_charges, 2)))

        predict_btn = st.button("🔮 Predict Churn Risk")

    with col_result:
        st.markdown("### 📊 Prediction Result")

        if predict_btn:
            input_dict = {
                'gender': gender, 'SeniorCitizen': senior,
                'Partner': partner, 'Dependents': dependents,
                'tenure': tenure, 'PhoneService': phone_service,
                'MultipleLines': multiple_lines, 'InternetService': internet,
                'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
                'DeviceProtection': device_protect, 'TechSupport': tech_support,
                'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
                'Contract': contract, 'PaperlessBilling': paperless,
                'PaymentMethod': payment, 'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
            }

            X_input = preprocess_input(input_dict)
            champion = get_champion()
            prob = champion.predict_proba(X_input)[0][1]
            pred = int(prob >= 0.5)

            st.markdown("---")
            if pred == 1:
                st.markdown(f'<div class="churn-high">⚠️ HIGH CHURN RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="churn-low">✅ RETAINED ACCOUNT</div>', unsafe_allow_html=True)

            st.markdown(f"""
            <div style="margin-top:1.5rem;">
                <div style="background:#1a1a3e;border-radius:10px;padding:1.2rem;border:1px solid #3a3a6a;">
                    <div style="color:#aaa;font-size:0.85rem;text-transform:uppercase;letter-spacing:1px;">Confidence Score</div>
                    <div style="font-size:3rem;font-weight:700;color:{'#ff6b6b' if pred else '#00d4ff'};">{prob:.1%}</div>
                    <div style="color:#aaa;font-size:0.85rem;">Probability of Churn</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            risk_level = "🔴 Critical" if prob > 0.75 else "🟠 High" if prob > 0.5 else "🟡 Moderate" if prob > 0.35 else "🟢 Low"
            st.markdown(f"**Risk Level:** {risk_level}")

            st.markdown("---")
            st.markdown("**Key Risk Factors:**")
            risk_factors = []
            if contract == "Month-to-month": risk_factors.append("📋 Month-to-month contract")
            if internet == "Fiber optic":    risk_factors.append("🌐 Fiber optic (high churn service)")
            if tenure < 12:                  risk_factors.append("📅 Short tenure (<12 months)")
            if monthly_charges > 70:         risk_factors.append("💰 High monthly charges")
            if online_security == "No":      risk_factors.append("🔒 No online security")
            if payment == "Electronic check":risk_factors.append("💳 Electronic check payment")

            if risk_factors:
                for rf in risk_factors:
                    st.markdown(f"• {rf}")
            else:
                st.markdown("• No major risk indicators detected")

            st.markdown("---")
            st.markdown("**Model:** XGBoost (Tuned) | AUC: 0.838")
        else:
            st.info("👈 Fill in customer details and click **Predict Churn Risk**")
            st.markdown("""
            **How it works:**
            1. Enter customer profile details
            2. Click the prediction button
            3. Get instant churn probability
            4. Review key risk factors

            **Model:** XGBoost with GridSearchCV tuning
            - Trained on 5,625 customer records
            - Validated on 1,407 test records
            - ROC-AUC: **0.838**
            """)
