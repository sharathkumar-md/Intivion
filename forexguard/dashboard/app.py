"""
Streamlit dashboard for ForexGuard.

Four pages:
  Overview        — KPI cards, alert tier distribution, model comparison, IF vs LSTM scatter
  Alert List      — filterable/paginated table of flagged users with progress bars
  User Detail     — deep dive on a single user: SHAP bar chart, LSTM reconstruction error chart,
                    graph risk metrics, LLM narrative, ground truth label (if available)
  Score Distribution — histograms and box plots by tier, ROC curves for all three models

Run with:
  streamlit run forexguard/dashboard/app.py
  Score Distribution : histograms by tier
"""

import os
import sys
from pathlib import Path

# Make sure the repo root is on the path so imports work on Streamlit Cloud
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "ForexGuard | Anomaly Detection",
    page_icon   = "🛡️",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# Generate data if running on a fresh deployment (e.g. Streamlit Community Cloud)
from forexguard.dashboard.startup import ensure_data
ensure_data()

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent.parent
RAW_DIR    = BASE_DIR / "data" / "raw"

# ──────────────────────────────────────────────────────────────────────────────
# Data loading (cached)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_data():
    final_scores = pd.read_parquet(RAW_DIR / "final_scores.parquet").reset_index()
    alerts       = pd.read_parquet(RAW_DIR / "alerts.parquet").reset_index()
    graph_feats  = pd.read_parquet(RAW_DIR / "features_graph.parquet").reset_index()

    llm_path = RAW_DIR / "llm_summaries.parquet"
    llm_summaries = pd.read_parquet(llm_path).reset_index() if llm_path.exists() else pd.DataFrame()

    labels_path = RAW_DIR / "labels.parquet"
    labels = (
        pd.read_parquet(labels_path)
        .groupby("user_id")[["is_anomaly", "anomaly_type"]]
        .agg({"is_anomaly": "max", "anomaly_type": lambda x: ", ".join(x.unique())})
        .reset_index()
        if labels_path.exists() else pd.DataFrame()
    )
    return final_scores, alerts, graph_feats, llm_summaries, labels


final_scores, alerts, graph_feats, llm_summaries, labels = load_data()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/ForexGuard-v1.0-blue", use_container_width=False)
    st.title("ForexGuard")
    st.caption("Real-Time Anomaly Detection Engine")
    st.divider()

    page = st.radio(
        "Navigation",
        ["Overview", "Alert List", "User Detail", "Score Distribution"],
    )

    st.divider()
    st.metric("Total Users",    f"{len(final_scores):,}")
    st.metric("Active Alerts",  f"{len(alerts):,}")
    st.metric("Critical Flags", f"{(final_scores['alert_tier']=='CRITICAL').sum():,}")


# ──────────────────────────────────────────────────────────────────────────────
# Helper colours
# ──────────────────────────────────────────────────────────────────────────────

TIER_COLORS = {
    "CRITICAL": "#FF4B4B",
    "HIGH":     "#FF8C00",
    "MEDIUM":   "#FFD700",
    "LOW":      "#00C49A",
}


# ──────────────────────────────────────────────────────────────────────────────
# Page: Overview
# ──────────────────────────────────────────────────────────────────────────────

if page == "Overview":
    st.title("🛡️ ForexGuard — Anomaly Detection Overview")

    # KPI row
    col1, col2, col3, col4, col5 = st.columns(5)
    tier_counts = final_scores["alert_tier"].value_counts()
    col1.metric("CRITICAL",   tier_counts.get("CRITICAL", 0), delta=None)
    col2.metric("HIGH",       tier_counts.get("HIGH", 0))
    col3.metric("MEDIUM",     tier_counts.get("MEDIUM", 0))
    col4.metric("LOW (clean)",tier_counts.get("LOW", 0))
    col5.metric("AUROC (IF)", "0.9554", delta="+0.03 vs baseline")

    st.divider()

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Alert Tier Distribution")
        tier_df = tier_counts.reset_index()
        tier_df.columns = ["Tier", "Count"]
        tier_df["Color"] = tier_df["Tier"].map(TIER_COLORS)
        fig = px.pie(
            tier_df, values="Count", names="Tier",
            color="Tier", color_discrete_map=TIER_COLORS,
            hole=0.4,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.subheader("Model Performance Comparison")
        metrics_df = pd.DataFrame({
            "Model":  ["Isolation Forest", "LSTM Autoencoder", "Ensemble (max-fusion)"],
            "AUROC":  [0.9554, 0.9159, 0.9531],
            "AUPRC":  [0.8276, 0.7334, 0.8349],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="AUROC", x=metrics_df["Model"], y=metrics_df["AUROC"],
                              marker_color="#4A90D9"))
        fig2.add_trace(go.Bar(name="AUPRC", x=metrics_df["Model"], y=metrics_df["AUPRC"],
                              marker_color="#7B68EE"))
        fig2.update_layout(barmode="group", yaxis_range=[0.6, 1.0],
                           legend=dict(orientation="h"))
        st.plotly_chart(fig2, use_container_width=True)

    # Score scatter
    st.subheader("IF Score vs LSTM Score (all users)")
    plot_df = final_scores[["user_id", "if_score", "lstm_score", "alert_tier"]].copy()
    fig3 = px.scatter(
        plot_df, x="if_score", y="lstm_score", color="alert_tier",
        color_discrete_map=TIER_COLORS,
        opacity=0.6, size_max=6,
        labels={"if_score": "Isolation Forest Score", "lstm_score": "LSTM Score"},
        hover_data=["user_id"],
    )
    fig3.add_hline(y=0.5, line_dash="dot", line_color="gray", annotation_text="LSTM threshold")
    fig3.add_vline(x=0.5, line_dash="dot", line_color="gray", annotation_text="IF threshold")
    st.plotly_chart(fig3, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# Page: Alert List
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Alert List":
    st.title("🚨 Alert Queue")

    col1, col2 = st.columns([1, 3])
    with col1:
        tier_filter = st.multiselect(
            "Filter by Tier",
            options=["CRITICAL", "HIGH", "MEDIUM"],
            default=["CRITICAL", "HIGH"],
        )
        score_min = st.slider("Min Composite Score", 0.0, 1.0, 0.40, 0.05)
        top_n     = st.number_input("Show top N", min_value=10, max_value=500, value=50, step=10)

    filtered = alerts.copy()
    if tier_filter:
        filtered = filtered[filtered["alert_tier"].isin(tier_filter)]
    filtered = filtered[filtered["composite_score"] >= score_min]
    filtered = filtered.sort_values("composite_score", ascending=False).head(top_n)

    with col2:
        st.caption(f"Showing {len(filtered)} alerts")

    # Colour-coded table
    def _tier_color(tier):
        colors = {"CRITICAL": "background-color: #FFE4E4",
                  "HIGH":     "background-color: #FFF3E0",
                  "MEDIUM":   "background-color: #FFFDE7",
                  "LOW":      ""}
        return colors.get(tier, "")

    display_cols = ["user_id", "composite_score", "if_score", "lstm_score", "alert_tier", "explanation"]
    display_df = filtered[[c for c in display_cols if c in filtered.columns]].copy()
    display_df["composite_score"] = display_df["composite_score"].round(4)
    display_df["if_score"]        = display_df["if_score"].round(4)
    display_df["lstm_score"]      = display_df["lstm_score"].round(4)
    display_df["explanation"]     = display_df["explanation"].str[:120] + "..."

    st.dataframe(
        display_df,
        use_container_width=True,
        height=600,
        column_config={
            "composite_score": st.column_config.ProgressColumn(
                "Composite", min_value=0, max_value=1, format="%.4f"
            ),
            "if_score": st.column_config.ProgressColumn(
                "IF Score", min_value=0, max_value=1, format="%.4f"
            ),
            "lstm_score": st.column_config.ProgressColumn(
                "LSTM Score", min_value=0, max_value=1, format="%.4f"
            ),
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page: User Detail
# ──────────────────────────────────────────────────────────────────────────────

elif page == "User Detail":
    st.title("🔍 User Detail View")

    # User selector
    all_users = final_scores["user_id"].tolist()
    alert_users = alerts["user_id"].tolist() if "user_id" in alerts.columns else []
    default_user = alert_users[0] if alert_users else all_users[0]

    selected_user = st.selectbox(
        "Select User ID",
        options=all_users,
        index=all_users.index(default_user) if default_user in all_users else 0,
    )

    user_row = final_scores[final_scores["user_id"] == selected_user].iloc[0]

    # Header
    tier = user_row.get("alert_tier", "LOW")
    tier_color = TIER_COLORS.get(tier, "#888")
    st.markdown(
        f"<h3 style='color:{tier_color}'>User: {selected_user} | Alert Tier: {tier}</h3>",
        unsafe_allow_html=True,
    )

    # Score KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Composite Score", f"{user_row.get('composite_score', 0):.4f}")
    c2.metric("IF Score",        f"{user_row.get('if_score', 0):.4f}")
    c3.metric("LSTM Score",      f"{user_row.get('lstm_score', 0):.4f}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # SHAP explanation
        st.subheader("Isolation Forest — SHAP Top Features")
        shap_feats = str(user_row.get("shap_top_features", ""))
        shap_vals  = str(user_row.get("shap_top_values", ""))
        if shap_feats and shap_feats != "0":
            feat_list = [f.strip() for f in shap_feats.split(",")]
            val_list  = [float(v.strip()) for v in shap_vals.split(",") if v.strip()]
            if feat_list and val_list:
                shap_df = pd.DataFrame({"Feature": feat_list[:len(val_list)], "SHAP Value": val_list})
                shap_df = shap_df.sort_values("SHAP Value")
                fig_shap = px.bar(
                    shap_df, x="SHAP Value", y="Feature", orientation="h",
                    color="SHAP Value", color_continuous_scale=["green", "white", "red"],
                    title="SHAP Values (negative = pushes toward anomaly)",
                )
                st.plotly_chart(fig_shap, use_container_width=True)
        else:
            st.info("No SHAP data available for this user.")

    with col_right:
        # LSTM feature errors
        st.subheader("LSTM Autoencoder — Reconstruction Errors")
        lstm_feats = str(user_row.get("lstm_top_features", ""))
        lstm_errs  = str(user_row.get("lstm_top_errors", ""))
        if lstm_feats and lstm_feats != "0":
            feat_list = [f.strip() for f in lstm_feats.split(",")]
            err_list  = [float(v.strip()) for v in lstm_errs.split(",") if v.strip()]
            if feat_list and err_list:
                lstm_df = pd.DataFrame({"Feature": feat_list[:len(err_list)], "Reconstruction MSE": err_list})
                fig_lstm = px.bar(
                    lstm_df, x="Reconstruction MSE", y="Feature", orientation="h",
                    color="Reconstruction MSE",
                    color_continuous_scale=["white", "orange", "red"],
                    title="Per-Feature Reconstruction Error",
                )
                st.plotly_chart(fig_lstm, use_container_width=True)
        else:
            st.info("No LSTM reconstruction data for this user.")

    # Graph features
    st.subheader("Graph Risk Features")
    graph_row = graph_feats[graph_feats["user_id"] == selected_user]
    if not graph_row.empty:
        gr = graph_row.iloc[0]
        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Shared IP Users",  int(gr.get("shared_ip_user_count", 0)))
        gc2.metric("Component Size",   int(gr.get("component_size", 1)))
        gc3.metric("Community Size",   int(gr.get("community_size", 1)))
        gc4.metric("Hub Neighbor",     "YES" if gr.get("is_hub_neighbor", 0) else "NO")

    # LLM Summary
    st.subheader("LLM Risk Narrative")
    if not llm_summaries.empty and "user_id" in llm_summaries.columns:
        user_llm = llm_summaries[llm_summaries["user_id"] == selected_user]
        if not user_llm.empty:
            summary_text = user_llm.iloc[0]["llm_summary"]
            st.info(summary_text)
        else:
            st.caption("No LLM summary for this user (only generated for HIGH/CRITICAL).")
            st.info(user_row.get("explanation", "No explanation available."))
    else:
        st.info(user_row.get("explanation", "No explanation available."))

    # Ground truth (if available)
    if not labels.empty and "user_id" in labels.columns:
        label_row = labels[labels["user_id"] == selected_user]
        if not label_row.empty:
            lr = label_row.iloc[0]
            is_anom = int(lr.get("is_anomaly", 0))
            anom_type = str(lr.get("anomaly_type", "normal"))
            if is_anom:
                st.warning(f"Ground truth: ANOMALOUS — type(s): {anom_type}")
            else:
                st.success("Ground truth: NORMAL user")


# ──────────────────────────────────────────────────────────────────────────────
# Page: Score Distribution
# ──────────────────────────────────────────────────────────────────────────────

elif page == "Score Distribution":
    st.title("📊 Score Distributions")

    score_col = st.selectbox(
        "Select score column",
        ["composite_score", "if_score", "lstm_score", "graph_risk_score"],
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader(f"{score_col} — Full Distribution")
        fig = px.histogram(
            final_scores, x=score_col, color="alert_tier",
            nbins=50, barmode="overlay",
            color_discrete_map=TIER_COLORS,
            opacity=0.7,
        )
        fig.add_vline(x=0.40, line_dash="dash", line_color="gold",   annotation_text="MEDIUM")
        fig.add_vline(x=0.60, line_dash="dash", line_color="orange",  annotation_text="HIGH")
        fig.add_vline(x=0.80, line_dash="dash", line_color="red",    annotation_text="CRITICAL")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Box Plot by Tier")
        fig2 = px.box(
            final_scores, x="alert_tier", y=score_col,
            color="alert_tier", color_discrete_map=TIER_COLORS,
            category_orders={"alert_tier": ["LOW", "MEDIUM", "HIGH", "CRITICAL"]},
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ROC curve (approximation using sorted scores)
    if not labels.empty and "user_id" in labels.columns:
        st.subheader("ROC Curves")
        try:
            from sklearn.metrics import roc_curve

            merged = final_scores.merge(labels[["user_id", "is_anomaly"]], on="user_id", how="inner")
            y_true = merged["is_anomaly"].values

            fig_roc = go.Figure()
            fig_roc.add_shape(type="line", line=dict(dash="dash", color="gray"),
                              x0=0, x1=1, y0=0, y1=1)

            for col_name, color, name in [
                ("if_score",        "#4A90D9", "Isolation Forest"),
                ("lstm_score",      "#7B68EE", "LSTM Autoencoder"),
                ("composite_score", "#FF6B6B", "Ensemble"),
            ]:
                fpr, tpr, _ = roc_curve(y_true, merged[col_name].values)
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=name,
                                             line=dict(color=color, width=2)))

            fig_roc.update_layout(
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                xaxis=dict(range=[0, 1]),
                yaxis=dict(range=[0, 1]),
                legend=dict(x=0.6, y=0.2),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        except Exception as exc:
            st.warning(f"Could not plot ROC curve: {exc}")
