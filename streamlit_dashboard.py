"""
Professional Streamlit Dashboard for PDRM Malaysia Asset & Facility Monitoring System
Features: Descriptive Analytics, Predictive Analytics, and AI Report Generator

Author: PDRM Dashboard Team
Date: August 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime, timedelta
import os
import openai
import io
from docx import Document
from docx.shared import Inches
from fpdf import FPDF
from dotenv import load_dotenv
import openai
from docx import Document
from fpdf import FPDF
import io
import warnings

warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="PDRM Asset & Facility Monitoring Dashboard",
    page_icon="🚔",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for professional styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .kpi-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1f4e79;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    """Load the PDRM asset dataset."""
    try:
        df = pd.read_csv("assets_data.csv")
        df["purchase_date"] = pd.to_datetime(df["purchase_date"])
        df["last_maintenance_date"] = pd.to_datetime(df["last_maintenance_date"])
        df["next_maintenance_due"] = pd.to_datetime(df["next_maintenance_due"])
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


@st.cache_resource
def load_model():
    """Load the trained ML model and related components."""
    try:
        model = joblib.load("enhanced_pdrm_asset_condition_model.joblib")
        label_encoder = joblib.load("enhanced_label_encoder.joblib")

        with open("enhanced_model_info.json", "r") as f:
            model_info = json.load(f)

        return model, label_encoder, model_info
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None


def calculate_kpis(df):
    """Calculate key performance indicators."""
    total_assets = len(df)
    good_assets = len(df[df["condition"] == "Good"])
    needs_action = len(df[df["condition"] == "Needs Action"])
    damaged_assets = len(df[df["condition"] == "Damaged"])

    # Calculate percentages
    good_pct = (good_assets / total_assets) * 100
    needs_action_pct = (needs_action / total_assets) * 100
    damaged_pct = (damaged_assets / total_assets) * 100

    # Calculate overdue maintenance
    overdue_maintenance = len(df[df["next_maintenance_due"] < datetime.now()])

    return {
        "total_assets": total_assets,
        "good_assets": good_assets,
        "needs_action": needs_action,
        "damaged_assets": damaged_assets,
        "good_pct": good_pct,
        "needs_action_pct": needs_action_pct,
        "damaged_pct": damaged_pct,
        "overdue_maintenance": overdue_maintenance,
    }


def display_kpis(kpis):
    """Display KPI cards in a professional layout."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="metric-value">{kpis['total_assets']:,}</div>
            <div class="metric-label">Total Assets</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="metric-value" style="color: #28a745;">{kpis['good_pct']:.1f}%</div>
            <div class="metric-label">Assets in Good Condition</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="metric-value" style="color: #ffc107;">{kpis['needs_action_pct']:.1f}%</div>
            <div class="metric-label">Need Maintenance</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""
        <div class="kpi-card">
            <div class="metric-value" style="color: #dc3545;">{kpis['damaged_pct']:.1f}%</div>
            <div class="metric-label">Damaged Assets</div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def create_trend_analysis(df):
    """Create trend analysis chart."""
    # Extract year from purchase date
    df["purchase_year"] = df["purchase_date"].dt.year

    # Group by year and condition
    trend_data = df.groupby(["purchase_year", "condition"]).size().unstack(fill_value=0)
    trend_data = trend_data.reset_index()

    fig = px.line(
        trend_data.melt(
            id_vars=["purchase_year"], var_name="Condition", value_name="Count"
        ),
        x="purchase_year",
        y="Count",
        color="Condition",
        title="Asset Condition Trends Over Years",
        labels={"purchase_year": "Year", "Count": "Number of Assets"},
    )

    fig.update_layout(height=500, showlegend=True, title_font_size=16, title_x=0.5)

    return fig


def create_distribution_chart(df):
    """Create distribution charts by asset type and condition."""
    # Asset type distribution
    type_condition = df.groupby(["category", "condition"]).size().unstack(fill_value=0)

    fig = px.bar(
        type_condition.reset_index().melt(
            id_vars=["category"], var_name="Condition", value_name="Count"
        ),
        x="category",
        y="Count",
        color="Condition",
        title="Asset Distribution by Category and Condition",
        labels={"category": "Asset Category", "Count": "Number of Assets"},
    )

    fig.update_layout(height=500, title_font_size=16, title_x=0.5)

    return fig


def create_heatmap(df):
    """Create heatmap of asset conditions by state."""
    # Create pivot table for heatmap
    heatmap_data = df.groupby(["state", "condition"]).size().unstack(fill_value=0)

    fig = px.imshow(
        heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        color_continuous_scale="RdYlGn_r",
        title="Asset Condition Distribution by State",
        labels=dict(x="Condition", y="State", color="Count"),
    )

    fig.update_layout(height=600, title_font_size=16, title_x=0.5)

    return fig


def create_detailed_category_breakdown(df):
    """Create detailed breakdown charts for each category showing asset types and conditions."""
    category_charts = {}

    for category in df["category"].unique():
        category_df = df[df["category"] == category]

        # Count assets by type and condition
        type_condition = (
            category_df.groupby(["type", "condition"]).size().reset_index(name="count")
        )

        # Create stacked bar chart for this category
        fig = px.bar(
            type_condition,
            x="type",
            y="count",
            color="condition",
            title=f"{category} - Asset Types and Conditions",
            labels={"type": "Asset Type", "count": "Number of Assets"},
            color_discrete_map={
                "Good": "#2E8B57",  # Sea Green
                "Needs Action": "#FF8C00",  # Dark Orange
                "Damaged": "#DC143C",  # Crimson
            },
        )

        # Rotate x-axis labels for better readability
        fig.update_layout(
            height=400,
            title_font_size=14,
            xaxis_tickangle=-45,
            xaxis_title_font_size=12,
            yaxis_title_font_size=12,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        category_charts[category] = fig

    return category_charts


def create_category_summary_metrics(df):
    """Create summary metrics for each category."""
    category_metrics = {}

    for category in df["category"].unique():
        category_df = df[df["category"] == category]

        total_assets = len(category_df)
        good_assets = len(category_df[category_df["condition"] == "Good"])
        needs_action = len(category_df[category_df["condition"] == "Needs Action"])
        damaged_assets = len(category_df[category_df["condition"] == "Damaged"])

        # Calculate percentages
        good_pct = (good_assets / total_assets * 100) if total_assets > 0 else 0
        needs_action_pct = (
            (needs_action / total_assets * 100) if total_assets > 0 else 0
        )
        damaged_pct = (damaged_assets / total_assets * 100) if total_assets > 0 else 0

        # Get top asset types
        top_types = category_df["type"].value_counts().head(3)

        category_metrics[category] = {
            "total_assets": total_assets,
            "good_assets": good_assets,
            "good_pct": good_pct,
            "needs_action": needs_action,
            "needs_action_pct": needs_action_pct,
            "damaged_assets": damaged_assets,
            "damaged_pct": damaged_pct,
            "top_types": top_types.to_dict(),
        }

    return category_metrics


def create_maintenance_priority_chart(df):
    """Create chart showing maintenance priority by asset type."""
    # Calculate maintenance urgency score
    today = datetime.now()
    df_copy = df.copy()
    df_copy["next_maintenance_due"] = pd.to_datetime(df_copy["next_maintenance_due"])
    df_copy["days_to_maintenance"] = (df_copy["next_maintenance_due"] - today).dt.days

    # Priority scoring: overdue gets highest priority
    df_copy["priority_score"] = df_copy["days_to_maintenance"].apply(
        lambda x: 100 if x < 0 else (90 if x < 30 else (70 if x < 90 else 30))
    )

    # Group by category and type, get average priority
    priority_data = (
        df_copy.groupby(["category", "type"])
        .agg({"priority_score": "mean", "asset_id": "count"})
        .rename(columns={"asset_id": "asset_count"})
        .reset_index()
    )

    # Create priority chart
    fig = px.scatter(
        priority_data,
        x="asset_count",
        y="priority_score",
        color="category",
        size="asset_count",
        hover_data=["type"],
        title="Maintenance Priority by Asset Type",
        labels={
            "asset_count": "Number of Assets",
            "priority_score": "Maintenance Priority Score",
            "category": "Category",
        },
    )

    fig.update_layout(height=500, title_font_size=16, title_x=0.5)
    return fig


def create_maintenance_forecast_by_category(df, months_ahead=6):
    """Create detailed maintenance demand forecast by category and type."""
    today = datetime.now()

    # Create forecast data for each category
    categories = df["category"].unique()
    forecast_data = []

    for category in categories:
        category_df = df[df["category"] == category]

        for month in range(1, months_ahead + 1):
            future_date = today + timedelta(days=month * 30)

            # Assets due for maintenance in this month
            due_this_month = category_df[
                (
                    category_df["next_maintenance_due"]
                    >= today + timedelta(days=(month - 1) * 30)
                )
                & (category_df["next_maintenance_due"] <= future_date)
            ]

            # Assets likely to need maintenance based on usage patterns
            high_usage = category_df[
                (category_df["usage_hours"] > category_df["usage_hours"].quantile(0.8))
                | (category_df["mileage_km"] > category_df["mileage_km"].quantile(0.8))
            ]

            # Calculate aging assets (approaching high risk)
            aging_assets = category_df[
                ((today - category_df["purchase_date"]).dt.days / 365.25)
                > 7  # Over 7 years old
            ]

            forecast_data.append(
                {
                    "Month": f"Month {month}",
                    "Category": category,
                    "Scheduled_Maintenance": len(due_this_month),
                    "High_Usage_Risk": len(high_usage) // 12,  # Spread over year
                    "Aging_Assets_Risk": len(aging_assets) // 24,  # Spread over 2 years
                    "Total_Demand": len(due_this_month)
                    + (len(high_usage) // 12)
                    + (len(aging_assets) // 24),
                }
            )

    return pd.DataFrame(forecast_data)


def create_forecast_charts(df, months_ahead=6):
    """Create multiple forecast charts for different categories."""
    forecast_df = create_maintenance_forecast_by_category(df, months_ahead)

    # 1. Stacked Bar Chart for Total Maintenance Demand
    fig1 = px.bar(
        forecast_df,
        x="Month",
        y="Total_Demand",
        color="Category",
        title=f"Total Maintenance Demand Forecast - Next {months_ahead} Months",
        labels={"Total_Demand": "Assets Requiring Maintenance"},
        color_discrete_map={
            "Weapons": "#FF6B6B",
            "ICT Assets": "#4ECDC4",
            "Vehicles": "#45B7D1",
            "Devices": "#FFA07A",
        },
    )
    fig1.update_layout(height=500, title_font_size=16, title_x=0.5)

    # 2. Line Chart for Category-wise Trends
    fig2 = px.line(
        forecast_df,
        x="Month",
        y="Total_Demand",
        color="Category",
        title="Maintenance Demand Trends by Category",
        markers=True,
        color_discrete_map={
            "Weapons": "#FF6B6B",
            "ICT Assets": "#4ECDC4",
            "Vehicles": "#45B7D1",
            "Devices": "#FFA07A",
        },
    )
    fig2.update_layout(height=400, title_font_size=16, title_x=0.5)

    # 3. Sunburst Chart for Risk Breakdown
    risk_data = []
    for category in df["category"].unique():
        category_total = forecast_df[forecast_df["Category"] == category][
            "Total_Demand"
        ].sum()
        scheduled = forecast_df[forecast_df["Category"] == category][
            "Scheduled_Maintenance"
        ].sum()
        high_usage = forecast_df[forecast_df["Category"] == category][
            "High_Usage_Risk"
        ].sum()
        aging = forecast_df[forecast_df["Category"] == category][
            "Aging_Assets_Risk"
        ].sum()

        risk_data.extend(
            [
                {
                    "Category": category,
                    "Risk_Type": "Scheduled",
                    "Value": scheduled,
                    "Parent": category,
                },
                {
                    "Category": category,
                    "Risk_Type": "High Usage",
                    "Value": high_usage,
                    "Parent": category,
                },
                {
                    "Category": category,
                    "Risk_Type": "Aging Assets",
                    "Value": aging,
                    "Parent": category,
                },
            ]
        )

    # Add category totals
    for category in df["category"].unique():
        total = forecast_df[forecast_df["Category"] == category]["Total_Demand"].sum()
        risk_data.append(
            {"Category": category, "Risk_Type": category, "Value": total, "Parent": ""}
        )

    risk_df = pd.DataFrame(risk_data)

    fig3 = px.sunburst(
        risk_df,
        path=["Category", "Risk_Type"],
        values="Value",
        title="Maintenance Risk Breakdown by Category",
        color_discrete_map={
            "Weapons": "#FF6B6B",
            "ICT Assets": "#4ECDC4",
            "Vehicles": "#45B7D1",
            "Devices": "#FFA07A",
        },
    )
    fig3.update_layout(height=500, title_font_size=16, title_x=0.5)

    return fig1, fig2, fig3, forecast_df


def create_asset_type_forecast(df, category, months_ahead=6):
    """Create detailed forecast for specific asset types within a category."""
    category_df = df[df["category"] == category]
    today = datetime.now()

    types_data = []
    for asset_type in category_df["type"].unique():
        type_df = category_df[category_df["type"] == asset_type]

        for month in range(1, months_ahead + 1):
            future_date = today + timedelta(days=month * 30)

            # Calculate various risk factors
            due_maintenance = len(
                type_df[
                    (
                        type_df["next_maintenance_due"]
                        >= today + timedelta(days=(month - 1) * 30)
                    )
                    & (type_df["next_maintenance_due"] <= future_date)
                ]
            )

            high_usage = (
                len(
                    type_df[
                        type_df["usage_hours"] > type_df["usage_hours"].quantile(0.75)
                    ]
                )
                // 12
            )
            aging = (
                len(type_df[((today - type_df["purchase_date"]).dt.days / 365.25) > 6])
                // 24
            )

            types_data.append(
                {
                    "Month": month,
                    "Type": asset_type,
                    "Due_Maintenance": due_maintenance,
                    "High_Usage_Risk": high_usage,
                    "Aging_Risk": aging,
                    "Total_Risk": due_maintenance + high_usage + aging,
                }
            )

    types_df = pd.DataFrame(types_data)

    fig = px.bar(
        types_df,
        x="Month",
        y="Total_Risk",
        color="Type",
        title=f"{category} - Maintenance Demand by Asset Type",
        labels={"Total_Risk": "Assets at Risk", "Month": "Months from Now"},
    )
    fig.update_layout(height=400, title_font_size=14, title_x=0.5)

    return fig, types_df


def generate_insights_and_recommendations(df, forecast_df):
    """Generate insights and recommendations based on forecast data."""
    insights = []
    recommendations = []

    # Analyze peak demand periods
    peak_month = forecast_df.groupby("Month")["Total_Demand"].sum().idxmax()
    peak_demand = forecast_df.groupby("Month")["Total_Demand"].sum().max()

    insights.append(
        f"📈 **Peak Maintenance Period**: {peak_month} with {peak_demand} assets requiring attention"
    )

    # Category analysis
    category_demand = (
        forecast_df.groupby("Category")["Total_Demand"]
        .sum()
        .sort_values(ascending=False)
    )
    highest_demand_category = category_demand.index[0]
    highest_demand_value = category_demand.iloc[0]

    insights.append(
        f"🔧 **Highest Demand Category**: {highest_demand_category} ({highest_demand_value} assets)"
    )

    # Risk analysis
    total_scheduled = forecast_df["Scheduled_Maintenance"].sum()
    total_high_usage = forecast_df["High_Usage_Risk"].sum()
    total_aging = forecast_df["Aging_Assets_Risk"].sum()

    insights.append(
        f"📊 **Risk Breakdown**: {total_scheduled} scheduled, {total_high_usage} high-usage risk, {total_aging} aging assets"
    )

    # Generate recommendations
    recommendations.extend(
        [
            f"🎯 **Immediate Action**: Prepare maintenance capacity for {highest_demand_category} assets",
            f"📅 **Resource Planning**: Allocate extra resources for {peak_month}",
            "🔍 **Proactive Monitoring**: Focus on high-usage assets to prevent unexpected failures",
            "📋 **Inventory Management**: Stock spare parts for aging equipment",
            "👥 **Staff Planning**: Schedule additional maintenance personnel during peak periods",
            "💰 **Budget Allocation**: Reserve budget for emergency repairs and replacements",
        ]
    )

    # Specific category recommendations
    if highest_demand_category == "Weapons":
        recommendations.append(
            "⚠️ **Critical Priority**: Weapons maintenance affects operational readiness"
        )
    elif highest_demand_category == "Vehicles":
        recommendations.append(
            "🚗 **Fleet Management**: Consider vehicle rotation to reduce maintenance burden"
        )
    elif highest_demand_category == "ICT Assets":
        recommendations.append(
            "💻 **Technology Refresh**: Plan for hardware upgrades to reduce maintenance needs"
        )
    elif highest_demand_category == "Devices":
        recommendations.append(
            "📱 **Device Lifecycle**: Implement device replacement cycles"
        )

    return insights, recommendations


def preprocess_prediction_data(input_data):
    """Preprocess input data for prediction."""
    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Convert dates
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    df["last_maintenance_date"] = pd.to_datetime(df["last_maintenance_date"])
    df["next_maintenance_due"] = pd.to_datetime(df["next_maintenance_due"])

    # Calculate derived features
    df["asset_age_days"] = (datetime.now() - df["purchase_date"]).dt.days
    df["asset_age_years"] = df["asset_age_days"] / 365.25
    df["maintenance_frequency"] = (datetime.now() - df["last_maintenance_date"]).dt.days
    df["days_to_maintenance"] = (df["next_maintenance_due"] - datetime.now()).dt.days

    # Usage intensity
    df["usage_intensity"] = df["usage_hours"] / (df["asset_age_years"] + 0.1)
    df["annual_mileage"] = df["mileage_km"] / (df["asset_age_years"] + 0.1)

    # Flags and categories
    df["maintenance_overdue"] = (df["days_to_maintenance"] < 0).astype(int)
    df["high_usage_hours"] = (df["usage_hours"] > 3000).astype(int)
    df["high_mileage"] = (df["mileage_km"] > 75000).astype(int)

    # Age category
    df["age_category"] = pd.cut(
        df["asset_age_years"],
        bins=[0, 2, 5, 8, float("inf")],
        labels=["New", "Young", "Mature", "Old"],
    )

    # Location and asset type risks
    df["high_risk_location"] = (
        df["state"].isin(["Sabah", "Sarawak", "Kelantan"]).astype(int)
    )
    critical_types = [
        "M4 Carbine",
        "M16A4",
        "HK416",
        "Windows Server 2019",
        "Linux RHEL Server",
    ]
    df["critical_asset"] = df["type"].isin(critical_types).astype(int)

    # Category-specific features
    df["has_ammunition_data"] = (df["ammunition_count"].notna()).astype(int)
    df["ammunition_count"] = df["ammunition_count"].fillna(0)

    if "warranty_expiry" in df.columns and df["warranty_expiry"].iloc[0]:
        warranty_date = pd.to_datetime(df["warranty_expiry"])
        df["warranty_expired"] = (warranty_date < datetime.now()).astype(int)
    else:
        df["warranty_expired"] = 0

    # Fill missing values
    df["usage_hours"] = df["usage_hours"].fillna(0)
    df["mileage_km"] = df["mileage_km"].fillna(0)
    df["assigned_officer"] = df["assigned_officer"].fillna("Unknown")

    # Select features for prediction
    feature_columns = [
        "category",
        "type",
        "state",
        "city",
        "assigned_officer",
        "asset_age_years",
        "maintenance_frequency",
        "days_to_maintenance",
        "usage_hours",
        "usage_intensity",
        "mileage_km",
        "annual_mileage",
        "age_category",
        "high_risk_location",
        "critical_asset",
        "maintenance_overdue",
        "high_usage_hours",
        "high_mileage",
        "has_ammunition_data",
        "ammunition_count",
        "warranty_expired",
    ]

    return df[feature_columns]


def generate_ai_report(report_config, kpis, df):
    """Generate AI-powered report using OpenAI."""
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Prepare data summary for the AI
        data_summary = f"""
        PDRM Asset & Facility Monitoring Report Data:
        
        KEY PERFORMANCE INDICATORS:
        - Total Assets: {kpis['total_assets']:,}
        - Assets in Good Condition: {kpis['good_assets']:,} ({kpis['good_pct']:.1f}%)
        - Assets Needing Action: {kpis['needs_action']:,} ({kpis['needs_action_pct']:.1f}%)
        - Damaged Assets: {kpis['damaged_assets']:,} ({kpis['damaged_pct']:.1f}%)
        - Assets with Overdue Maintenance: {kpis['overdue_maintenance']:,}
        
        ASSET DISTRIBUTION:
        {df.groupby(['category', 'condition']).size().to_string()}
        
        STATE-WISE DISTRIBUTION:
        {df.groupby(['state', 'condition']).size().head(20).to_string()}
        """

        prompt = f"""
        You are a professional asset management analyst for the Royal Malaysia Police (PDRM). 
        Generate a comprehensive executive report based on the following data:
        
        {data_summary}
        
        Please include the following sections in your report:
        1. EXECUTIVE SUMMARY
        2. KEY FINDINGS AND INSIGHTS
        3. ASSET CONDITION ANALYSIS
        4. MAINTENANCE RECOMMENDATIONS
        5. RISK ASSESSMENT
        6. STRATEGIC RECOMMENDATIONS
        
        The report should be professional, actionable, and suitable for senior management decision-making.
        Focus on operational efficiency, cost optimization, and asset reliability.
        """

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional asset management analyst.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=2000,
            temperature=0.7,
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error generating AI report: {str(e)}"


def export_report_to_word_with_charts(
    report_content, charts_data, df, filename="PDRM_Asset_Report.docx"
):
    """Export comprehensive report to Word document with charts."""
    try:
        doc = Document()

        # Add title
        title = doc.add_heading("PDRM Asset & Facility Monitoring Report", 0)
        title.alignment = 1  # Center alignment

        # Add metadata
        doc.add_paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        doc.add_paragraph("Royal Malaysia Police - Asset Management Division")
        doc.add_page_break()

        # Add executive summary
        doc.add_heading("EXECUTIVE SUMMARY", level=1)

        # Add KPIs table
        doc.add_heading("Key Performance Indicators", level=2)
        table = doc.add_table(rows=1, cols=2)
        table.style = "Table Grid"

        # Add table headers
        hdr_cells = table.rows[0].cells
        hdr_cells[0].text = "Metric"
        hdr_cells[1].text = "Value"

        # Add KPI data
        kpi_data = [
            ("Total Assets", f"{charts_data['total_assets']:,}"),
            (
                "Assets in Good Condition",
                f"{charts_data['good_assets']:,} ({charts_data['good_pct']:.1f}%)",
            ),
            (
                "Assets Needing Action",
                f"{charts_data['needs_action']:,} ({charts_data['needs_action_pct']:.1f}%)",
            ),
            (
                "Damaged Assets",
                f"{charts_data['damaged_assets']:,} ({charts_data['damaged_pct']:.1f}%)",
            ),
            ("Overdue Maintenance", f"{charts_data['overdue_maintenance']:,}"),
        ]

        for metric, value in kpi_data:
            row_cells = table.add_row().cells
            row_cells[0].text = metric
            row_cells[1].text = value

        doc.add_paragraph()

        # Generate and add charts
        doc.add_heading("VISUAL ANALYTICS", level=1)

        # 1. Asset Distribution Chart
        doc.add_heading("Asset Distribution by Category and Condition", level=2)

        try:
            # Create distribution chart
            type_condition = (
                df.groupby(["category", "condition"]).size().unstack(fill_value=0)
            )
            fig_dist = px.bar(
                type_condition.reset_index().melt(
                    id_vars=["category"], var_name="Condition", value_name="Count"
                ),
                x="category",
                y="Count",
                color="Condition",
                title="Asset Distribution by Category and Condition",
                labels={"category": "Asset Category", "Count": "Number of Assets"},
                color_discrete_map={
                    "Good": "#2E8B57",
                    "Needs Action": "#FFD700",
                    "Damaged": "#DC143C",
                },
            )
            fig_dist.update_layout(height=400, title_font_size=14, title_x=0.5)

            # Save chart as image
            img_bytes = fig_dist.to_image(format="png", width=800, height=400)
            img_stream = io.BytesIO(img_bytes)
            doc.add_picture(img_stream, width=Inches(6))
            doc.add_paragraph()

        except Exception as e:
            doc.add_paragraph(f"Chart generation error: {str(e)}")

        # 2. Asset Condition Pie Chart
        doc.add_heading("Overall Asset Condition Distribution", level=2)

        try:
            condition_counts = df["condition"].value_counts()
            fig_pie = px.pie(
                values=condition_counts.values,
                names=condition_counts.index,
                title="Overall Asset Condition Distribution",
                color_discrete_map={
                    "Good": "#2E8B57",
                    "Needs Action": "#FFD700",
                    "Damaged": "#DC143C",
                },
            )
            fig_pie.update_layout(height=400, title_font_size=14, title_x=0.5)

            # Save chart as image
            img_bytes = fig_pie.to_image(format="png", width=800, height=400)
            img_stream = io.BytesIO(img_bytes)
            doc.add_picture(img_stream, width=Inches(6))
            doc.add_paragraph()

        except Exception as e:
            doc.add_paragraph(f"Chart generation error: {str(e)}")

        # 3. State-wise Heatmap (simplified table for Word)
        doc.add_heading("Asset Distribution by State", level=2)

        try:
            state_condition = (
                df.groupby(["state", "condition"]).size().unstack(fill_value=0)
            )

            # Create table for state data
            state_table = doc.add_table(rows=1, cols=4)
            state_table.style = "Table Grid"

            # Headers
            headers = state_table.rows[0].cells
            headers[0].text = "State"
            headers[1].text = "Good"
            headers[2].text = "Needs Action"
            headers[3].text = "Damaged"

            # Add top 10 states data
            for state in state_condition.head(10).index:
                row_cells = state_table.add_row().cells
                row_cells[0].text = state
                row_cells[1].text = str(state_condition.loc[state].get("Good", 0))
                row_cells[2].text = str(
                    state_condition.loc[state].get("Needs Action", 0)
                )
                row_cells[3].text = str(state_condition.loc[state].get("Damaged", 0))

            doc.add_paragraph()

        except Exception as e:
            doc.add_paragraph(f"Table generation error: {str(e)}")

        # Add AI-generated content
        doc.add_heading("DETAILED ANALYSIS", level=1)

        # Split AI content into paragraphs
        paragraphs = report_content.split("\n\n")
        for paragraph in paragraphs:
            if paragraph.strip():
                # Check if it's a heading (starts with number or all caps)
                if paragraph.strip().isupper() or (
                    paragraph.strip() and paragraph.strip()[0].isdigit()
                ):
                    doc.add_heading(paragraph.strip(), level=2)
                else:
                    doc.add_paragraph(paragraph.strip())

        # Add category breakdown section
        doc.add_heading("ASSET CATEGORY BREAKDOWN", level=1)

        categories = ["Weapons", "ICT Assets", "Vehicles", "Devices"]
        for category in categories:
            if category in charts_data["category_stats"]:
                stats = charts_data["category_stats"][category]
                doc.add_heading(f"{category}", level=2)

                # Add category statistics
                doc.add_paragraph(f"Total Assets: {stats['total_assets']:,}")
                doc.add_paragraph(
                    f"Good Condition: {stats['good_count']:,} ({stats['good_pct']:.1f}%)"
                )
                doc.add_paragraph(
                    f"Needs Action: {stats['needs_action_count']:,} ({stats['needs_action_pct']:.1f}%)"
                )
                doc.add_paragraph(
                    f"Damaged: {stats['damaged_count']:,} ({stats['damaged_pct']:.1f}%)"
                )
                doc.add_paragraph(f"Average Age: {stats['avg_age_years']:.1f} years")

                # Add top asset types
                doc.add_paragraph("Top Asset Types:")
                for asset_type, count in stats["top_types"].head(5).items():
                    doc.add_paragraph(
                        f"• {asset_type}: {count:,} units", style="List Bullet"
                    )

                # Add category-specific chart
                try:
                    category_df = df[df["category"] == category]
                    type_condition_cat = (
                        category_df.groupby(["type", "condition"])
                        .size()
                        .reset_index(name="count")
                    )

                    if not type_condition_cat.empty:
                        fig_cat = px.bar(
                            type_condition_cat,
                            x="type",
                            y="count",
                            color="condition",
                            title=f"{category} - Asset Types and Conditions",
                            labels={
                                "type": "Asset Type",
                                "count": "Number of Assets",
                                "condition": "Condition",
                            },
                            color_discrete_map={
                                "Good": "#2E8B57",
                                "Needs Action": "#FFD700",
                                "Damaged": "#DC143C",
                            },
                        )
                        fig_cat.update_layout(
                            height=300,
                            title_font_size=12,
                            title_x=0.5,
                            xaxis={"tickangle": 45},
                        )

                        # Save category chart as image
                        img_bytes = fig_cat.to_image(
                            format="png", width=800, height=300
                        )
                        img_stream = io.BytesIO(img_bytes)
                        doc.add_picture(img_stream, width=Inches(6))

                except Exception as e:
                    doc.add_paragraph(f"Category chart error: {str(e)}")

                doc.add_paragraph()

        # Add recommendations
        doc.add_heading("STRATEGIC RECOMMENDATIONS", level=1)

        recommendations = [
            "Prioritize maintenance for overdue assets to prevent further deterioration",
            "Implement predictive maintenance schedule based on asset age and usage patterns",
            "Consider replacement program for assets with consistently poor condition ratings",
            "Enhance monitoring systems for high-risk asset categories",
            "Establish preventive maintenance protocols for critical equipment",
            "Allocate budget for high-maintenance categories (Weapons and Vehicles)",
            "Implement asset tracking system for better maintenance scheduling",
            "Train staff on proper asset care and maintenance procedures",
        ]

        for rec in recommendations:
            doc.add_paragraph(rec, style="List Bullet")

        # Add footer
        doc.add_page_break()
        doc.add_heading("REPORT METADATA", level=1)
        doc.add_paragraph(
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        doc.add_paragraph(f"Total Records Analyzed: {len(df):,}")
        doc.add_paragraph(
            f"Data Period: {df['purchase_date'].min()} to {df['purchase_date'].max()}"
        )
        doc.add_paragraph("Generated by: PDRM Asset Management Dashboard")
        doc.add_paragraph("Classification: Internal Use")

        # Save to memory
        doc_buffer = io.BytesIO()
        doc.save(doc_buffer)
        doc_buffer.seek(0)

        return doc_buffer.getvalue()

    except Exception as e:
        st.error(f"Error creating Word document: {str(e)}")
        return None


def export_report_to_pdf(report_content, filename="PDRM_Asset_Report.pdf"):
    """Export report to PDF."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "PDRM Asset & Facility Monitoring Report", ln=True, align="C")
    pdf.ln(10)

    # Add report content
    pdf.set_font("Arial", size=10)
    lines = report_content.split("\n")
    for line in lines:
        if len(line) > 80:
            # Split long lines
            words = line.split(" ")
            current_line = ""
            for word in words:
                if len(current_line + word) < 80:
                    current_line += word + " "
                else:
                    pdf.cell(
                        0,
                        6,
                        current_line.encode("latin-1", "replace").decode("latin-1"),
                        ln=True,
                    )
                    current_line = word + " "
            if current_line:
                pdf.cell(
                    0,
                    6,
                    current_line.encode("latin-1", "replace").decode("latin-1"),
                    ln=True,
                )
        else:
            pdf.cell(0, 6, line.encode("latin-1", "replace").decode("latin-1"), ln=True)

    return pdf.output(dest="S").encode("latin-1")


def main():
    """Main dashboard application."""

    # Header
    st.markdown(
        '<div class="main-header">🚔 PDRM Asset & Facility Monitoring Dashboard</div>',
        unsafe_allow_html=True,
    )

    # Load data
    df = load_data()
    if df is None:
        st.error("Unable to load data. Please check if 'assets_data.csv' exists.")
        return

    # Sidebar filters
    st.sidebar.header("🔍 Filters")

    # Asset type filter
    asset_types = ["All"] + list(df["category"].unique())
    selected_type = st.sidebar.selectbox("Asset Type", asset_types)

    # State filter
    states = ["All"] + list(df["state"].unique())
    selected_state = st.sidebar.selectbox("State", states)

    # Year range filter
    min_year = int(df["purchase_date"].dt.year.min())
    max_year = int(df["purchase_date"].dt.year.max())
    selected_years = st.sidebar.slider(
        "Purchase Year Range", min_year, max_year, (min_year, max_year)
    )

    # Apply filters
    filtered_df = df.copy()
    if selected_type != "All":
        filtered_df = filtered_df[filtered_df["category"] == selected_type]
    if selected_state != "All":
        filtered_df = filtered_df[filtered_df["state"] == selected_state]

    filtered_df = filtered_df[
        (filtered_df["purchase_date"].dt.year >= selected_years[0])
        & (filtered_df["purchase_date"].dt.year <= selected_years[1])
    ]

    # Calculate KPIs
    kpis = calculate_kpis(filtered_df)

    # Create tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "📊 Descriptive Analytics",
            "🔮 Predictive Analytics",
            "📝 AI Report Generator",
        ]
    )

    with tab1:
        st.header("📊 Descriptive Analytics")

        # Display KPIs
        display_kpis(kpis)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            # Trend analysis
            fig_trend = create_trend_analysis(filtered_df)
            st.plotly_chart(fig_trend, use_container_width=True)

            # Simple maintenance timeline
            today = datetime.now()
            upcoming_maintenance = filtered_df[
                (filtered_df["next_maintenance_due"] >= today)
                & (filtered_df["next_maintenance_due"] <= today + timedelta(days=90))
            ]

            fig_upcoming = px.histogram(
                upcoming_maintenance,
                x="category",
                title="Assets Due for Maintenance (Next 3 Months)",
                labels={"category": "Category", "count": "Number of Assets"},
            )
            fig_upcoming.update_layout(height=400, title_font_size=14)
            st.plotly_chart(fig_upcoming, use_container_width=True)

        with col2:
            # Distribution chart
            fig_dist = create_distribution_chart(filtered_df)
            st.plotly_chart(fig_dist, use_container_width=True)

            # Asset condition pie chart
            condition_counts = filtered_df["condition"].value_counts()
            fig_pie = px.pie(
                values=condition_counts.values,
                names=condition_counts.index,
                title="Overall Asset Condition Distribution",
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)

        # Heatmap
        st.subheader("🗺️ Geographic Distribution")
        fig_heatmap = create_heatmap(filtered_df)
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Detailed Category Breakdown Section
        st.subheader("🔍 Detailed Category Analysis")

        # Category summary metrics
        category_metrics = create_category_summary_metrics(filtered_df)

        # Display category metrics in columns
        cols = st.columns(len(category_metrics))
        for i, (category, metrics) in enumerate(category_metrics.items()):
            with cols[i]:
                st.metric(
                    label=f"{category} Total",
                    value=f"{metrics['total_assets']:,}",
                    help=f"Total {category.lower()} assets",
                )
                st.write(f"✅ Good: {metrics['good_pct']:.1f}%")
                st.write(f"⚠️ Needs Action: {metrics['needs_action_pct']:.1f}%")
                st.write(f"❌ Damaged: {metrics['damaged_pct']:.1f}%")

        # Detailed breakdown charts for each category
        st.subheader("📊 Asset Type Breakdown by Category")
        category_charts = create_detailed_category_breakdown(filtered_df)

        # Create tabs for each category
        if len(category_charts) > 0:
            category_tabs = st.tabs(list(category_charts.keys()))
            for i, (category, fig) in enumerate(category_charts.items()):
                with category_tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)

                    # Show top asset types for this category
                    category_df = filtered_df[filtered_df["category"] == category]
                    top_types = category_df["type"].value_counts().head(5)

                    st.write("**Top Asset Types:**")
                    for asset_type, count in top_types.items():
                        percentage = (count / len(category_df)) * 100
                        st.write(
                            f"• {asset_type}: {count:,} assets ({percentage:.1f}%)"
                        )

        # Maintenance Priority Analysis
        st.subheader("⚡ Maintenance Priority Analysis")
        fig_priority = create_maintenance_priority_chart(filtered_df)
        st.plotly_chart(fig_priority, use_container_width=True)

        # Priority explanation
        st.info(
            """
        **Priority Score Guide:**
        - 🔴 **90-100**: Overdue maintenance (immediate action required)
        - 🟠 **70-89**: Due within 30 days (schedule soon) 
        - 🟡 **50-69**: Due within 90 days (plan ahead)
        - 🟢 **30-49**: Long-term planning (3+ months)
        """
        )

    with tab2:
        st.header("🔮 Predictive Analytics - Maintenance Demand Forecasting")

        # Load model for display info
        model, label_encoder, model_info = load_model()

        # Model performance info
        if model_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model Type", model_info["model_name"])
            with col2:
                st.metric("Accuracy", f"{model_info['metrics']['accuracy']:.2%}")
            with col3:
                st.metric("F1-Score", f"{model_info['metrics']['f1_score']:.3f}")

        st.markdown("---")

        # Forecast period selector
        col1, col2 = st.columns([1, 3])
        with col1:
            forecast_months = st.selectbox(
                "Forecast Period",
                [3, 6, 9, 12],
                index=1,
                help="Select how many months ahead to forecast",
            )

        st.markdown("### 📈 Maintenance Demand Forecast")

        # Generate forecast charts
        fig1, fig2, fig3, forecast_df = create_forecast_charts(
            filtered_df, forecast_months
        )

        # Display main forecast charts
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)

        # Risk breakdown chart
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 🔍 Category-Specific Forecasts")

        # Category tabs for detailed analysis
        category_tabs = st.tabs(list(filtered_df["category"].unique()))

        for i, category in enumerate(filtered_df["category"].unique()):
            with category_tabs[i]:
                fig_cat, types_df = create_asset_type_forecast(
                    filtered_df, category, forecast_months
                )
                st.plotly_chart(fig_cat, use_container_width=True)

                # Top asset types needing attention
                top_types = (
                    types_df.groupby("Type")["Total_Risk"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(5)
                )

                st.markdown(f"**Top 5 {category} Types Requiring Attention:**")
                for idx, (asset_type, risk_score) in enumerate(top_types.items(), 1):
                    st.markdown(
                        f"{idx}. **{asset_type}**: {int(risk_score)} assets at risk"
                    )

        st.markdown("---")

        # Generate insights and recommendations
        st.markdown("### 💡 Insights & Recommendations")

        insights, recommendations = generate_insights_and_recommendations(
            filtered_df, forecast_df
        )

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 📊 Key Insights")
            for insight in insights:
                st.markdown(insight)

            # Additional statistical insights
            st.markdown("#### 📈 Statistical Summary")
            total_assets_at_risk = forecast_df["Total_Demand"].sum()
            avg_monthly_demand = (
                forecast_df.groupby("Month")["Total_Demand"].sum().mean()
            )

            st.markdown(
                f"- **Total Assets at Risk**: {int(total_assets_at_risk)} over {forecast_months} months"
            )
            st.markdown(
                f"- **Average Monthly Demand**: {int(avg_monthly_demand)} assets"
            )
            st.markdown(
                f"- **Peak vs Average Ratio**: {forecast_df.groupby('Month')['Total_Demand'].sum().max() / avg_monthly_demand:.1f}x"
            )

        with col2:
            st.markdown("#### 🎯 Action Recommendations")
            for recommendation in recommendations:
                st.markdown(recommendation)

            # Priority action items
            st.markdown("#### ⚡ Priority Actions")
            priority_items = [
                "📅 Schedule maintenance teams for peak periods",
                "📦 Order spare parts for high-risk asset types",
                "👥 Train additional maintenance personnel",
                "💰 Allocate emergency maintenance budget",
                "🔄 Implement preventive maintenance cycles",
            ]

            for item in priority_items:
                st.markdown(f"- {item}")

        # Forecast summary table
        st.markdown("### 📋 Forecast Summary Table")

        summary_table = (
            forecast_df.groupby(["Month", "Category"])
            .agg(
                {
                    "Scheduled_Maintenance": "sum",
                    "High_Usage_Risk": "sum",
                    "Aging_Assets_Risk": "sum",
                    "Total_Demand": "sum",
                }
            )
            .reset_index()
        )

        # Pivot for better display
        pivot_table = summary_table.pivot(
            index="Month", columns="Category", values="Total_Demand"
        ).fillna(0)
        st.dataframe(pivot_table, use_container_width=True)

        # Export forecast data
        if st.button("📥 Export Forecast Data"):
            csv_data = summary_table.to_csv(index=False)
            st.download_button(
                label="Download Forecast CSV",
                data=csv_data,
                file_name=f"maintenance_forecast_{forecast_months}months.csv",
                mime="text/csv",
            )

    with tab3:
        st.header("🤖 AI Report Generator")

        st.markdown(
            """
        Generate comprehensive AI-powered insights and recommendations based on your asset data.
        """
        )

        # Report configuration
        st.subheader("⚙️ Report Configuration")

        # Configuration options outside form
        col1, col2 = st.columns(2)

        with col1:
            include_kpis = st.checkbox("Include KPIs", value=True)
            include_trends = st.checkbox("Include Trend Analysis", value=True)
            include_distribution = st.checkbox(
                "Include Distribution Charts", value=True
            )

        with col2:
            include_heatmap = st.checkbox("Include Geographic Heatmap", value=True)
            include_forecast = st.checkbox("Include Maintenance Forecast", value=True)
            report_format = st.selectbox(
                "Report Format", ["Word Document", "PDF", "Text"], index=0
            )

        report_focus = st.selectbox(
            "Report Focus",
            [
                "Executive Summary",
                "Operational Analysis",
                "Strategic Planning",
                "Risk Assessment",
            ],
        )

        st.markdown("---")

        # Generate button - moved outside of form to fix the error
        if st.button("🚀 Generate AI Report", type="primary", use_container_width=True):
            report_config = {
                "include_kpis": include_kpis,
                "include_trends": include_trends,
                "include_distribution": include_distribution,
                "include_heatmap": include_heatmap,
                "include_forecast": include_forecast,
                "focus": report_focus,
            }

            with st.spinner("Generating AI-powered report... This may take a moment."):
                # Prepare charts data for Word export
                charts_data = {
                    "total_assets": len(filtered_df),
                    "good_assets": len(filtered_df[filtered_df["condition"] == "Good"]),
                    "needs_action": len(
                        filtered_df[filtered_df["condition"] == "Needs Action"]
                    ),
                    "damaged_assets": len(
                        filtered_df[filtered_df["condition"] == "Damaged"]
                    ),
                    "overdue_maintenance": len(
                        filtered_df[
                            pd.to_datetime(filtered_df["next_maintenance_due"])
                            < datetime.now()
                        ]
                    ),
                    "good_pct": (
                        len(filtered_df[filtered_df["condition"] == "Good"])
                        / len(filtered_df)
                    )
                    * 100,
                    "needs_action_pct": (
                        len(filtered_df[filtered_df["condition"] == "Needs Action"])
                        / len(filtered_df)
                    )
                    * 100,
                    "damaged_pct": (
                        len(filtered_df[filtered_df["condition"] == "Damaged"])
                        / len(filtered_df)
                    )
                    * 100,
                    "category_stats": {},
                }

                # Add category statistics
                for category in filtered_df["category"].unique():
                    cat_data = filtered_df[filtered_df["category"] == category].copy()
                    cat_data["asset_age_years"] = (
                        pd.to_datetime("today")
                        - pd.to_datetime(cat_data["purchase_date"])
                    ).dt.days / 365.25

                    charts_data["category_stats"][category] = {
                        "total_assets": len(cat_data),
                        "good_count": len(cat_data[cat_data["condition"] == "Good"]),
                        "needs_action_count": len(
                            cat_data[cat_data["condition"] == "Needs Action"]
                        ),
                        "damaged_count": len(
                            cat_data[cat_data["condition"] == "Damaged"]
                        ),
                        "good_pct": (
                            len(cat_data[cat_data["condition"] == "Good"])
                            / len(cat_data)
                        )
                        * 100,
                        "needs_action_pct": (
                            len(cat_data[cat_data["condition"] == "Needs Action"])
                            / len(cat_data)
                        )
                        * 100,
                        "damaged_pct": (
                            len(cat_data[cat_data["condition"] == "Damaged"])
                            / len(cat_data)
                        )
                        * 100,
                        "avg_age_years": cat_data["asset_age_years"].mean(),
                        "top_types": cat_data["type"].value_counts(),
                    }

                # Generate the report
                report_content = generate_ai_report(report_config, kpis, filtered_df)

                if report_content:
                    st.success("✅ Report generated successfully!")

                    # Display the report
                    st.subheader("📄 Generated Report")
                    st.markdown(report_content)

                    st.markdown("---")
                    st.subheader("📥 Download Report")

                    # Download buttons
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        if report_format == "Word Document":
                            word_content = export_report_to_word_with_charts(
                                report_content, charts_data, df
                            )
                            if word_content:
                                st.download_button(
                                    label="📄 Download Word Report",
                                    data=word_content,
                                    file_name=f"PDRM_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx",
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                                    type="primary",
                                )
                        else:
                            st.info(
                                "Select Word Document format to download with charts"
                            )

                    with col2:
                        if report_format == "PDF":
                            pdf_content = export_report_to_pdf(report_content)
                            if pdf_content:
                                st.download_button(
                                    label="📄 Download PDF Report",
                                    data=pdf_content,
                                    file_name=f"PDRM_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                                    mime="application/pdf",
                                    type="primary",
                                )
                        else:
                            st.info("Select PDF format to download PDF")

                    with col3:
                        st.download_button(
                            label="📝 Download Text Report",
                            data=report_content,
                            file_name=f"PDRM_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                        )

                else:
                    st.error(
                        "Failed to generate report. Please check your OpenAI API key and try again."
                    )

        # Sample report preview
        st.markdown("---")
        st.subheader("📋 Sample Report Preview")

        with st.expander("View Sample Report Structure"):
            st.markdown(
                """
            **EXECUTIVE SUMMARY**
            - Asset overview and key metrics
            - Critical issues identification
            - High-priority recommendations
            
            **KEY FINDINGS AND INSIGHTS**
            - Category-wise condition analysis
            - Geographic distribution patterns
            - Age and maintenance correlations
            
            **ASSET CONDITION ANALYSIS**
            - Detailed condition breakdown by category
            - Risk factors and patterns
            - Performance indicators
            
            **MAINTENANCE RECOMMENDATIONS**
            - Overdue maintenance priorities
            - Predictive maintenance strategies
            - Resource allocation suggestions
            
            **RISK ASSESSMENT**
            - Critical asset identification
            - Failure probability analysis
            - Operational impact assessment
            
            **STRATEGIC RECOMMENDATIONS**
            - Investment priorities
            - Policy improvements
            - Long-term planning insights
            """
            )

        # AI Configuration
        with st.expander("🔧 Advanced Settings"):
            st.markdown("**AI Report Configuration**")

            col1, col2 = st.columns(2)
            with col1:
                st.info("**Report Focus Areas:**")
                st.write("• Executive Summary: High-level overview for leadership")
                st.write("• Operational Analysis: Day-to-day operational insights")
                st.write("• Strategic Planning: Long-term strategic recommendations")
                st.write("• Risk Assessment: Risk-focused analysis and mitigation")

            with col2:
                st.info("**Export Options:**")
                st.write("• Word Document: Full report with charts and formatting")
                st.write("• PDF: Professional document format")
                st.write("• Text: Plain text for further editing")

            st.warning(
                "**Note:** AI report generation requires a valid OpenAI API key. Set your API key as an environment variable 'OPENAI_API_KEY'."
            )

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>PDRM Asset & Facility Monitoring System | Developed for Royal Malaysia Police</p>
            <p>For support, contact: asset.management@pdrm.gov.my</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
