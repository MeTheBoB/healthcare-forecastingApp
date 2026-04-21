# My Final Year Project - NHS A&E Analytics Dashboard
# This script builds the interactive web dashboard using Streamlit.
# It loads the data, creates the sidebar filters, shows historical charts in tabs,
# and runs the AI models (Prophet or SARIMA) to forecast future attendances.

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from prophet import Prophet
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
import warnings

# Hide warning messages from the math libraries so the terminal stays clean
warnings.filterwarnings("ignore")


# Define the official NHS colours to use in the charts and CSS styling
NHS_BLUE   = "#005EB8"
NHS_DARK   = "#003087"
NHS_YELLOW = "#FFB81C"
NHS_GREEN  = "#009639"
NHS_RED    = "#DA291C"
NHS_LIGHT  = "#E8EDEE"


# Add custom CSS to make the dashboard look like an official NHS tool
# This changes the background color, font styles, and button colors.
CUSTOM_CSS = f"""
<style>

/* Import nice fonts for the text and headings */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* Set the main background to a light grey so the white cards stand out */
.stApp {{
    background: #F4F6F9;
    font-family: 'DM Sans', sans-serif;
}}

/* Make the dashboard wider and add some space at the top */
.block-container {{
    padding-top: 2rem !important;
    max-width: 1400px !important;
}}

/* Style for the big blue header banner at the top of the page */
.nh-header {{
    background: linear-gradient(135deg, {NHS_DARK} 0%, {NHS_BLUE} 60%, #0077CC 100%);
    border-radius: 16px;
    padding: 32px 40px 28px;
    margin-bottom: 24px;
}}
.nh-header h1 {{
    color: #FFFFFF !important;
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem !important;
    margin-bottom: 6px !important;
}}
.nh-header p {{
    color: rgba(255,255,255,0.78) !important;
    font-size: 1rem !important;
    margin: 0 !important;
}}

/* Style the white boxes (containers) around the charts and KPIs */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: #FFFFFF;
    border-radius: 12px !important;
    border: 1px solid #DDE3E9 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}}

/* Make the big KPI numbers NHS blue and bold - REDUCED FONT SIZE to fit large numbers */
[data-testid="stMetricValue"] > div {{
    color: {NHS_BLUE} !important;
    font-size: 1.15rem !important; 
    font-weight: 700 !important;
    white-space: nowrap !important;
    overflow: visible !important;
}}

/* Make the KPI titles small, grey, and uppercase - REDUCED FONT SIZE so titles fit */
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {{
    color: #4A5568 !important;
    font-size: 0.72rem !important; 
    font-weight: 600 !important;
    text-transform: uppercase;
    white-space: normal !important;
}}

/* Fix Tab Titles (Container Pages) to be visible and slightly larger */
button[data-baseweb="tab"] p {{
    color: #000000 !important; /* Make text black so it's visible */
    font-weight: 700 !important;
    font-size: 1.3rem !important;
}}
button[data-baseweb="tab"][aria-selected="true"] p {{
    color: {NHS_BLUE} !important; /* Make it NHS Blue when clicked/active */
}}

/* Style the chart headings (Making them bigger and Dark Blue as requested) */
h2, h3 {{
    color: {NHS_DARK} !important;
    font-family: 'DM Serif Display', serif !important;
}}
h2 {{
    font-size: 2.2rem !important;
}}
h3 {{
    font-size: 1.6rem !important;
    border-bottom: 2px solid {NHS_LIGHT};
    padding-bottom: 8px;
    margin-bottom: 16px !important;
}}

/* Style the left sidebar background to dark blue */
[data-testid="stSidebar"] {{
    background: {NHS_DARK} !important;
}}
[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
    background: {NHS_BLUE} !important;
}}

/* Change the main AI Forecast button to NHS Blue */
div.stButton > button[kind="primary"] {{
    background: {NHS_BLUE} !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}}
div.stButton > button[kind="primary"]:hover {{
    background: {NHS_DARK} !important;
}}

/* Change the timeline slider color to NHS Blue */
div[data-testid="stSlider"] div[role="slider"] {{
    background: {NHS_BLUE} !important;
}}

/* Make the info and success boxes look neat */
.stInfo {{
    border-left: 4px solid {NHS_BLUE} !important;
    border-radius: 8px !important;
}}
.stSuccess {{
    border-left: 4px solid {NHS_GREEN} !important;
    border-radius: 8px !important;
}}

/* Add a thin line to separate sections */
.section-divider {{
    border: none;
    border-top: 1px solid {NHS_LIGHT};
    margin: 24px 0;
}}

/* Custom progress bars for the regional performance tab */
.perf-bar-wrap {{
    background: #EEF2F5;
    border-radius: 99px;
    height: 12px;
    overflow: hidden;
    margin-top: 6px;
}}
.perf-bar-fill {{
    height: 100%;
    border-radius: 99px;
}}

/* Fix invisible text on the white Pills and Clear button in the sidebar */
[data-testid="stSidebar"] button:not([kind="primary"]),
[data-testid="stSidebar"] button:not([kind="primary"]) p,
[data-testid="stSidebar"] button:not([kind="primary"]) span,
[data-testid="stSidebar"] [data-testid="stPills"] * {{
    color: {NHS_DARK} !important;
    font-weight: 600 !important;
}}

</style>
"""

# Set up the main page layout and inject the CSS above
def setup_page():
    st.set_page_config(
        layout="wide",
        page_title="NHS A&E Analytics",
        initial_sidebar_state="expanded"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Load the data and clean it up.
# Using @st.cache_data means Streamlit only reads the CSV once, making the app much faster.
@st.cache_data
def load_data():
    try:
        df = pd.read_csv(r"C:\Users\DELL\Desktop\Uni25-26\Project\work\DATA\processedData\finalData.csv")
    except FileNotFoundError:
        st.error(r"C:\Users\DELL\Desktop\Uni25-26\Project\work\DATA\processedData\finalData.csv not found. Make sure the file exists at this path.")
        return pd.DataFrame()

    # Clean duplicates so we don't accidentally double-count patients
    df = df.drop_duplicates(subset=["MonthYear", "Organization"])

    # Fix the date column so Python knows it is a date
    df["MonthYear"] = pd.to_datetime(df["MonthYear"], errors="coerce")
    df = df.dropna(subset=["MonthYear"])

    # Filter data to start from January 2022 to avoid the extreme COVID lockdown anomalies
    df = df[df["MonthYear"] >= "2022-01-01"]
    df = df.sort_values("MonthYear").reset_index(drop=True)

    # Calculate the percentage of patients seen within 4 hours if it's missing
    if "Within_4h_%" not in df.columns:
        df["Within_4h_%"] = (df["Total Attendances < 4h"] / df["Total Attendances"] * 100).round(1)
    if "Over_4h_%" not in df.columns:
        df["Over_4h_%"] = 100 - df["Within_4h_%"]

    return df


# A small helper function to apply the same white background style to all our charts
def style_chart(chart):
    return (
        chart
        .configure(background="white")
        .configure_legend(labelColor="#333", titleColor="#333", labelFontSize=12)
        .configure_axis(labelColor="#555", titleColor="#555", gridColor="#EEF2F5", domainColor="#DDE3E9")
        .configure_view(strokeWidth=0)
    )


# Function to draw shaded background boxes on charts to show when COVID happened
def get_covid_bands(df):
    # Safety check: if we don't have this column, just return nothing so the app doesn't crash
    if "CovidPeriod" not in df.columns:
        return alt.layer()

    # Find the start and end dates for each COVID period
    period_ranges = (
        df.groupby("CovidPeriod")["MonthYear"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"CovidPeriod": "period", "min": "start", "max": "end"})
    )

    colour_map = {
        "Pre-COVID":    "#A8C5E0",
        "During COVID": "#FFD580",
        "Post-COVID":   "#A8D8B0"
    }

    # Draw a rectangle for each period
    bands = []
    for _, row in period_ranges.iterrows():
        colour = colour_map.get(row["period"], "#CCCCCC")
        band_data = pd.DataFrame({
            "start":  [row["start"]],
            "end":    [row["end"]],
            "period": [row["period"]]
        })
        band = (
            alt.Chart(band_data)
            .mark_rect(opacity=0.12, color=colour)
            .encode(
                x=alt.X("start:T"),
                x2=alt.X2("end:T")
            )
        )
        bands.append(band)

    return alt.layer(*bands)


# Build the side panel where the user selects their filters
def render_sidebar(df):
    with st.sidebar:
        # Title for the sidebar - Making this significantly larger as requested!
        st.markdown(
            """
            <div style='padding: 12px 0 20px'>
                <div style='font-family: "DM Serif Display", serif; font-size: 3.8rem; color: white; line-height: 1.1'>
                    NHS A&amp;E<br>Analytics
                </div>
                <div style='color: rgba(255,255,255,0.9); font-size: 1.6rem; font-weight: 500; margin-top: 10px'>
                    Powered by Prophet &amp; SARIMA
                </div>
            </div>
            <hr style='border-color: rgba(255,255,255,0.15); margin-bottom: 20px'>
            """,
            unsafe_allow_html=True
        )

        st.markdown("**Geography**")

        # Region dropdown
        region_options = ["All Regions"] + sorted(df["Region"].dropna().unique().tolist())
        selected_regions = st.multiselect(
            "Regions",
            options=region_options,
            default=["All Regions"],
            label_visibility="collapsed"
        )

        if "All Regions" not in selected_regions and len(selected_regions) > 0:
            df = df[df["Region"].isin(selected_regions)]

        # Hospital Trust dropdown (updates dynamically based on the region chosen above)
        trust_options = ["All Trusts"] + sorted(df["Organization"].dropna().unique().tolist())
        selected_trusts = st.multiselect(
            "Trusts",
            options=trust_options,
            default=["All Trusts"],
            label_visibility="collapsed"
        )

        if "All Trusts" not in selected_trusts and len(selected_trusts) > 0:
            df = df[df["Organization"].isin(selected_trusts)]

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0'>", unsafe_allow_html=True)

        # Timeline slider
        st.markdown("**Date Range**")
        min_date = df["MonthYear"].min().date()
        max_date = df["MonthYear"].max().date()

        start_date, end_date = st.slider(
            "Timeline",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="MMM YYYY",
            label_visibility="collapsed"
        )

        # Filter the data using the slider dates
        filtered_df = df[
            (df["MonthYear"].dt.date >= start_date) &
            (df["MonthYear"].dt.date <= end_date)
        ]

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0'>", unsafe_allow_html=True)

        # Buttons for the AI Forecast
        st.markdown("**AI Forecast**")
        forecast_options = [3, 4, 5, 6, 9, 12, 24]
        forecast_horizon = st.pills(
            "Horizon (months)",
            options=forecast_options,
            default=6,
            label_visibility="collapsed"
        )

        # Use session state memory so the forecast doesn't disappear when the user clicks an expander
        if "show_forecast" not in st.session_state:
            st.session_state.show_forecast = False

        # Place the Run and Clear buttons side by side
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Run AI Forecast", type="primary", use_container_width=True):
                st.session_state.show_forecast = True
        with btn_col2:
            if st.button("Clear Forecast", use_container_width=True):
                st.session_state.show_forecast = False
                
        # Grab the memory state to return to the main function
        run_forecast = st.session_state.show_forecast

        # Show a small summary text of how much data is selected
        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:20px 0 8px'>", unsafe_allow_html=True)
        n_trusts = filtered_df["Organization"].nunique()
        n_months = filtered_df["MonthYear"].nunique()
        
        st.markdown(
            f"<div style='color:rgba(255,255,255,0.5);font-size:0.75rem'>"
            f"{n_trusts:,} trusts selected over {n_months} months</div>",
            unsafe_allow_html=True
        )

    return filtered_df, forecast_horizon, run_forecast


# Draw the big numbers (KPIs) at the top of the screen
def render_kpis(df):
    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # Calculate total patients
    total_attendances = int(df["Total Attendances"].sum())
    total_admissions  = int(df["Total Emergency Admissions"].sum())

    # Calculate 4-hour performance percentage
    patients_within_4h = df["Total Attendances < 4h"].sum()
    if total_attendances > 0:
        four_hour_pct = patients_within_4h / total_attendances * 100
    else:
        four_hour_pct = 0

    # Calculate patients waiting on trolleys for beds
    over_4h  = int(df[">4h decision to admit"].sum())  if ">4h decision to admit"  in df.columns else None
    over_12h = int(df[">12h decision to admit"].sum()) if ">12h decision to admit" in df.columns else None
    other_admissions = int(df["Other Emergency admissions (i.e not via A&E)"].sum()) if "Other Emergency admissions (i.e not via A&E)" in df.columns else None

    # Decide how many columns we need based on what data is available
    num_cols = 6 if over_12h is not None else 3
    cols = st.columns(num_cols)

    with cols[0].container(border=True):
        st.metric(label="Total Attendances", value=f"{total_attendances:,}")

    with cols[1].container(border=True):
        st.metric(label="Emergency Admissions", value=f"{total_admissions:,}")

    with cols[2].container(border=True):
        # Add logic to show if they passed or failed the 78% NHS target
        if four_hour_pct >= 78:
            delta_text   = "Above 78% target"
            delta_colour = "normal"
        else:
            delta_text   = "Below 78% target"
            delta_colour = "inverse"

        st.metric(
            label="4-Hour Target Met",
            value=f"{four_hour_pct:.1f}%",
            delta=delta_text,
            delta_color=delta_colour
        )

    if over_4h is not None:
        with cols[3].container(border=True):
            st.metric(label=">4h Decision to Admit", value=f"{over_4h:,}")

    if over_12h is not None:
        with cols[4].container(border=True):
            st.metric(label=">12h Decision to Admit", value=f"{over_12h:,}")

    if other_admissions is not None:
        with cols[5].container(border=True):
            st.metric(label="Non-A&E Admissions", value=f"{other_admissions:,}")


# Draw the historical charts organised neatly into 4 tabs
def render_historical_charts(df):
    if df.empty:
        st.info("No data available.")
        return

    tab1, tab2, tab3, tab4 = st.tabs([
        "Attendances",
        "4-Hour Performance",
        "Regional Breakdown",
        "Admission Pathways"
    ])

    # Tab 1: Line chart of attendances over time
    with tab1:
        col_left, col_right = st.columns([3, 1])

        with col_left.container(border=True):
            st.markdown("### Monthly A&E Attendances")
            num_trusts = df["Organization"].nunique()

            # Smart grouping: if too many trusts are selected, combine them into one line
            # so the chart doesn't look like a messy hairball.
            if num_trusts > 5:
                st.info(
                    f"Smart Grouping Active: {num_trusts} trusts are selected. "
                    f"The chart has been aggregated into a single trendline to keep it readable."
                )
                time_df = df.groupby("MonthYear", as_index=False)[["Total Attendances"]].sum()

                line_chart = (
                    alt.Chart(time_df)
                    .mark_line(point=True, color=NHS_BLUE, strokeWidth=2.5)
                    .encode(
                        x=alt.X("MonthYear:T", title=None),
                        y=alt.Y("Total Attendances:Q", title="Attendances", axis=alt.Axis(format="~s")),
                        tooltip=[
                            alt.Tooltip("MonthYear:T", format="%b %Y"),
                            alt.Tooltip("Total Attendances:Q", format=",.0f")
                        ]
                    )
                )
                # HEIGHT set to 500 for a much larger, readable chart
                att_chart = style_chart((get_covid_bands(df) + line_chart).properties(height=500).interactive())

            elif num_trusts > 1:
                # Show separate lines for comparing a small number of trusts
                time_df = df.groupby(["MonthYear", "Organization"], as_index=False)[["Total Attendances"]].sum()

                line_chart = (
                    alt.Chart(time_df)
                    .mark_line(point=True, strokeWidth=2)
                    .encode(
                        x=alt.X("MonthYear:T", title=None),
                        y=alt.Y("Total Attendances:Q", title="Attendances", axis=alt.Axis(format="~s")),
                        color=alt.Color("Organization:N", legend=alt.Legend(title=None, orient="bottom", labelLimit=280)),
                        tooltip=[
                            "Organization:N",
                            alt.Tooltip("MonthYear:T", format="%b %Y"),
                            alt.Tooltip("Total Attendances:Q", format=",.0f")
                        ]
                    )
                )
                att_chart = style_chart((get_covid_bands(df) + line_chart).properties(height=500).interactive())

            else:
                # Just one trust selected
                time_df = df.groupby("MonthYear", as_index=False)[["Total Attendances"]].sum()

                line_chart = (
                    alt.Chart(time_df)
                    .mark_line(point=True, color=NHS_BLUE, strokeWidth=2.5)
                    .encode(
                        x=alt.X("MonthYear:T", title=None),
                        y=alt.Y("Total Attendances:Q", title="Attendances", axis=alt.Axis(format="~s")),
                        tooltip=[
                            alt.Tooltip("MonthYear:T", format="%b %Y"),
                            alt.Tooltip("Total Attendances:Q", format=",.0f")
                        ]
                    )
                )
                att_chart = style_chart((get_covid_bands(df) + line_chart).properties(height=500).interactive())

            st.altair_chart(att_chart, use_container_width=True, theme=None)

            if "CovidPeriod" in df.columns:
                st.caption("Pre-COVID | During COVID | Post-COVID (shaded background bands)")

        # Donut chart showing the different types of A&E departments
        with col_right.container(border=True):
            st.markdown("### Department Mix")

            type_data = pd.DataFrame({
                "Type":  ["Type 1 (Major)", "Type 2 (Single Spec)", "Type 3 (Minor/UTC)"],
                "Count": [df["Type 1"].sum(), df["Type 2"].sum(), df["Type 3"].sum()]
            })
            
            # Calculate percentages for the labels
            total_type = type_data["Count"].sum()
            type_data["Pct"] = type_data["Count"] / total_type if total_type > 0 else 0

            base = alt.Chart(type_data).encode(
                theta=alt.Theta("Count:Q", stack=True),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=type_data["Type"].tolist(),
                        range=[NHS_BLUE, NHS_YELLOW, NHS_GREEN]
                    ),
                    legend=alt.Legend(orient="bottom", title=None, labelLimit=200)
                ),
                tooltip=[
                    alt.Tooltip("Type:N"),
                    alt.Tooltip("Pct:Q", format=".1%", title="%"),
                    alt.Tooltip("Count:Q", format=",.0f")
                ]
            )

            # Donut radius matches the taller 500 height of the line chart
            arc   = base.mark_arc(innerRadius=70)
            # Display percentage directly on the chart
            label = base.mark_text(radius=115, fontSize=14, fontWeight="bold", fill="white").encode(
                text=alt.Text("Pct:Q", format=".0%")
            )

            st.altair_chart(style_chart((arc + label).properties(height=500)), use_container_width=True, theme=None)


    # Tab 2: 4-Hour wait times and performance
    with tab2:
        col_a, col_b = st.columns(2)

        with col_a.container(border=True):
            st.markdown("### Attendances vs 4-Hour Target")

            bar_data = df.groupby("MonthYear", as_index=False)[
                ["Total Attendances < 4h", "Total Attendances > 4 hours"]
            ].sum()

            bar_data = bar_data.rename(columns={
                "Total Attendances < 4h":       "Within 4h",
                "Total Attendances > 4 hours":  "Breached 4h"
            })

            bar_melted = bar_data.melt(id_vars="MonthYear", var_name="Category", value_name="Attendances")

            bar_chart = (
                alt.Chart(bar_melted)
                .mark_bar()
                .encode(
                    x=alt.X("MonthYear:T", title=None),
                    y=alt.Y("Attendances:Q", title="Attendances", axis=alt.Axis(format="~s")),
                    color=alt.Color(
                        "Category:N",
                        scale=alt.Scale(
                            domain=["Within 4h", "Breached 4h"],
                            range=[NHS_BLUE, NHS_YELLOW]
                        ),
                        legend=alt.Legend(orient="bottom", title=None)
                    ),
                    tooltip=[
                        alt.Tooltip("MonthYear:T", format="%b %Y"),
                        alt.Tooltip("Category:N"),
                        alt.Tooltip("Attendances:Q", format=",.0f")
                    ]
                )
                .properties(height=340)
                .interactive()
            )

            st.altair_chart(style_chart(bar_chart), use_container_width=True, theme=None)

        with col_b.container(border=True):
            st.markdown("### Type 1 Performance Over Time")

            if "Type1_Under4h_%" in df.columns:
                t1_data = df.groupby("MonthYear", as_index=False)["Type1_Under4h_%"].mean()

                perf_line = (
                    alt.Chart(t1_data)
                    .mark_line(point=True, color=NHS_BLUE, strokeWidth=2.5)
                    .encode(
                        x=alt.X("MonthYear:T", title=None),
                        y=alt.Y(
                            "Type1_Under4h_%:Q",
                            title="% Within 4h",
                            scale=alt.Scale(domain=[60, 100])
                        ),
                        tooltip=[
                            alt.Tooltip("MonthYear:T", format="%b %Y"),
                            alt.Tooltip("Type1_Under4h_%:Q", format=".1f", title="% Within 4h")
                        ]
                    )
                )

                # Draw a line across the chart to show the 78% target goal
                target_line = (
                    alt.Chart(pd.DataFrame({"y": [78.0]}))
                    .mark_rule(color=NHS_YELLOW, strokeDash=[4, 3], strokeWidth=1.8)
                    .encode(y=alt.Y("y:Q"))
                )

                target_label = (
                    alt.Chart(pd.DataFrame({
                        "x":    [t1_data["MonthYear"].max()],
                        "y":    [78.7],
                        "text": ["78% target (2025/26)"]
                    }))
                    .mark_text(align="right", fontSize=10, color="#B07800", fontWeight="bold")
                    .encode(
                        x=alt.X("x:T"),
                        y=alt.Y("y:Q"),
                        text="text:N"
                    )
                )

                combined = (perf_line + target_line + target_label).properties(height=340).interactive()
                st.altair_chart(style_chart(combined), use_container_width=True, theme=None)
                st.caption("Dashed line = 78% NHS England target for 2025/26")
            else:
                st.info("Type 1 performance column not found in the data.")

        # Area chart showing patients waiting on trolleys for beds
        if ">4h decision to admit" in df.columns:
            st.markdown("")
            with st.container(border=True):
                st.markdown("### Trolley Waits — Decision to Admit")

                trolley_data = df.groupby("MonthYear", as_index=False)[
                    [">4h decision to admit", ">12h decision to admit"]
                ].sum()

                trolley_melted = trolley_data.melt(
                    id_vars="MonthYear",
                    var_name="Wait Category",
                    value_name="Patients"
                )

                trolley_chart = (
                    alt.Chart(trolley_melted)
                    .mark_area(opacity=0.7, interpolate="monotone")
                    .encode(
                        x=alt.X("MonthYear:T", title=None),
                        y=alt.Y("Patients:Q", title="Patients", axis=alt.Axis(format="~s")),
                        color=alt.Color(
                            "Wait Category:N",
                            scale=alt.Scale(
                                domain=[">4h decision to admit", ">12h decision to admit"],
                                range=[NHS_YELLOW, NHS_RED]
                            ),
                            legend=alt.Legend(orient="bottom", title=None)
                        ),
                        tooltip=[
                            alt.Tooltip("MonthYear:T", format="%b %Y"),
                            alt.Tooltip("Wait Category:N"),
                            alt.Tooltip("Patients:Q", format=",.0f")
                        ]
                    )
                    .properties(height=280)
                    .interactive()
                )

                st.altair_chart(style_chart(trolley_chart), use_container_width=True, theme=None)
                st.caption("Trolley waits measure how many patients waited over 4h or 12h for a bed.")

    # Tab 3: Compare different regions against each other
    with tab3:
        col_r1, col_r2 = st.columns(2)

        with col_r1.container(border=True):
            st.markdown("### Attendances by Region")

            region_totals = (
                df.groupby("Region", as_index=False)["Total Attendances"]
                .sum()
                .sort_values("Total Attendances", ascending=False)
            )

            region_bar = (
                alt.Chart(region_totals)
                .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color=NHS_BLUE)
                .encode(
                    x=alt.X("Total Attendances:Q", title="Total Attendances", axis=alt.Axis(format="~s")),
                    y=alt.Y("Region:N", sort="-x", title=None),
                    tooltip=[
                        alt.Tooltip("Region:N"),
                        alt.Tooltip("Total Attendances:Q", format=",.0f")
                    ]
                )
                .properties(height=360)
            )
            st.altair_chart(style_chart(region_bar), use_container_width=True, theme=None)

        with col_r2.container(border=True):
            st.markdown("### 4-Hour Performance by Region")

            region_perf = (
                df.groupby("Region", as_index=False)
                .agg(
                    within_4h=("Total Attendances < 4h", "sum"),
                    total=("Total Attendances", "sum")
                )
            )
            region_perf["pct"] = region_perf["within_4h"] / region_perf["total"] * 100
            region_perf = region_perf.sort_values("pct", ascending=False)

            # Draw a custom progress bar for each region using HTML/CSS
            for _, row in region_perf.iterrows():
                pct = row["pct"]
                if pct >= 78:
                    bar_colour = NHS_GREEN
                elif pct >= 65:
                    bar_colour = NHS_YELLOW
                else:
                    bar_colour = NHS_RED

                st.markdown(
                    f"""
                    <div style='margin-bottom:10px'>
                        <div style='display:flex;justify-content:space-between;font-size:0.83rem;color:#333'>
                            <span>{row['Region']}</span>
                            <span style='font-weight:600;color:{bar_colour}'>{pct:.1f}%</span>
                        </div>
                        <div class='perf-bar-wrap'>
                            <div class='perf-bar-fill' style='width:{min(pct, 100):.1f}%;background:{bar_colour}'></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Tab 4: How did patients get admitted?
    with tab4:
        col_p1, col_p2 = st.columns([3, 2])

        with col_p1.container(border=True):
            st.markdown("### Emergency Admissions — via A&E vs Other")

            if "Other Emergency admissions (i.e not via A&E)" in df.columns:
                pathway_data = df.groupby("MonthYear", as_index=False).agg(
                    via_ae=("Total Emergency Admissions via A&E", "sum"),
                    other=("Other Emergency admissions (i.e not via A&E)", "sum")
                )
                pathway_data = pathway_data.rename(columns={"via_ae": "Via A&E", "other": "Other"})

                pathway_melted = pathway_data.melt(
                    id_vars="MonthYear",
                    var_name="Pathway",
                    value_name="Admissions"
                )

                pathway_chart = (
                    alt.Chart(pathway_melted)
                    .mark_area(interpolate="monotone", opacity=0.75)
                    .encode(
                        x=alt.X("MonthYear:T", title=None),
                        y=alt.Y("Admissions:Q", title="Admissions", axis=alt.Axis(format="~s"), stack="zero"),
                        color=alt.Color(
                            "Pathway:N",
                            scale=alt.Scale(
                                domain=["Via A&E", "Other"],
                                range=[NHS_BLUE, NHS_GREEN]
                            ),
                            legend=alt.Legend(orient="bottom", title=None)
                        ),
                        tooltip=[
                            alt.Tooltip("MonthYear:T", format="%b %Y"),
                            alt.Tooltip("Pathway:N"),
                            alt.Tooltip("Admissions:Q", format=",.0f")
                        ]
                    )
                    # INCREASED HEIGHT to 500 for a much larger, readable chart
                    .properties(height=500)
                    .interactive()
                )
                st.altair_chart(style_chart(pathway_chart), use_container_width=True, theme=None)

        with col_p2.container(border=True):
            st.markdown("### Admission Route Split")

            total_via_ae = float(df["Total Emergency Admissions via A&E"].sum())
            total_other  = float(df["Other Emergency admissions (i.e not via A&E)"].sum())
            grand_total  = total_via_ae + total_other

            if grand_total > 0:
                route_data = pd.DataFrame({
                    "Route":      ["Via A&E", "Other"],
                    "Admissions": [total_via_ae, total_other],
                    "Pct":        [total_via_ae / grand_total, total_other / grand_total] # Stored as fraction for percentage formatting
                })

                base2 = alt.Chart(route_data).encode(
                    theta=alt.Theta("Admissions:Q", stack=True),
                    color=alt.Color(
                        "Route:N",
                        scale=alt.Scale(domain=["Via A&E", "Other"], range=[NHS_BLUE, NHS_GREEN]),
                        legend=alt.Legend(orient="bottom", title=None)
                    ),
                    tooltip=[
                        alt.Tooltip("Route:N"),
                        alt.Tooltip("Pct:Q", format=".1%", title="%"),
                        alt.Tooltip("Admissions:Q", format=",.0f")
                    ]
                )

                # INCREASED donut radius so it matches the taller 500 height of the area chart
                arc2   = base2.mark_arc(innerRadius=70)
                # Display percentage directly on the chart
                label2 = base2.mark_text(radius=115, fontSize=14, fontWeight="bold", fill="white").encode(
                    text=alt.Text("Pct:Q", format=".0%")
                )

                st.altair_chart(
                    style_chart((arc2 + label2).properties(height=500)),
                    use_container_width=True,
                    theme=None
                )


# Run the AI forecasting models when the user clicks the button
def run_and_render_forecasts(df, horizon):

    # Make sure horizon is an integer so we don't get TypeErrors!
    if not horizon:
        horizon = 6
    horizon = int(horizon)

    # Group up the historical data so Prophet can read it easily
    train_data = (
        df.groupby("MonthYear", as_index=False)["Total Attendances"]
        .sum()
        .rename(columns={"MonthYear": "ds", "Total Attendances": "y"})
        .sort_values("ds")
    )

    # Check if we have enough data (we need at least 24 months to learn yearly patterns)
    if len(train_data) < 24:
        st.info("Not enough historical data. The model needs at least 24 months to detect yearly seasonality patterns. Please expand the date slider range.")
        st.session_state.show_forecast = False # Clear state to prevent looping bug
        return

    # Decide which algorithm to use: SARIMA for short-term, Prophet for long-term
    model_type = "SARIMA" if horizon <= 3 else "Prophet"

    # Show a loading spinner so the user knows the computer is doing math
    with st.spinner(f"Training {model_type} model — this may take a moment..."):

        if model_type == "SARIMA":
            # Set up the data for SARIMA
            ts = train_data.set_index("ds")["y"].asfreq("MS").ffill()

            sarima_model = sm.tsa.statespace.SARIMAX(
                ts,
                order=(1, 2, 1),
                seasonal_order=(1, 1, 1, 12),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

            # Predict into the future
            forecast_result = sarima_model.get_forecast(steps=horizon)

            forecast_df = pd.DataFrame({
                "Date": forecast_result.predicted_mean.index,
                "Predicted Attendances": forecast_result.predicted_mean.values.round(0).astype(int)
            })

            # Calculate errors on the historical data (skip first 13 months)
            y_true = ts.values[13:]
            y_pred = sarima_model.fittedvalues.values[13:]

        else:
            # Set up the Prophet model
            prophet_model = Prophet(interval_width=0.95, seasonality_mode="additive")
            prophet_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            prophet_model.fit(train_data)

            future_dates = prophet_model.make_future_dataframe(periods=horizon, freq="MS")
            prophet_output = prophet_model.predict(future_dates)

            # Keep only the future predictions
            future_only = prophet_output[prophet_output["ds"] > train_data["ds"].max()]

            forecast_df = pd.DataFrame({
                "Date": future_only["ds"],
                "Predicted Attendances": future_only["yhat"].round(0).astype(int)
            })

            # Get the historical errors
            in_sample = prophet_output[prophet_output["ds"] <= train_data["ds"].max()]
            y_true = train_data["y"].values
            y_pred = in_sample["yhat"].values

        # Stage 2: Linear Regression Cascade
        # Predict physical hospital admissions based on the attendances we just forecasted
        lr_data = df.groupby("MonthYear", as_index=False)[
            ["Total Attendances", "Total Emergency Admissions"]
        ].sum()

        lr_model = LinearRegression()
        lr_model.fit(lr_data[["Total Attendances"]], lr_data["Total Emergency Admissions"])

        forecast_df["Predicted Admissions"] = lr_model.predict(
            forecast_df[["Predicted Attendances"]].rename(
                columns={"Predicted Attendances": "Total Attendances"}
            )
        ).astype(int)

        # Work out the mathematical error scores
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse  = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    st.success(f"Model trained: {model_type} + Linear Regression cascade")

    # Display the accuracy metric cards
    st.markdown("### Model Accuracy (Historical Fit)")
    m1, m2, m3, m4 = st.columns(4)

    with m1.container(border=True):
        st.metric(label="MAPE — % Error", value=f"{mape:.2f}%")
    with m2.container(border=True):
        st.metric(label="MAE — Avg Patient Error", value=f"{mae:,.0f}")
    with m3.container(border=True):
        st.metric(label="RMSE — Penalty Error", value=f"{rmse:,.0f}")
    with m4.container(border=True):
        st.metric(label="MSE — Squared Error", value=f"{mse:,.0f}")

    # Add an expander so the user can read what the errors mean
    with st.expander("What do these metrics mean?"):
        st.markdown("""
| Metric | What it tells you |
|--------|------------------|
| **MAPE** | Average error as a percentage. The easiest way to explain model accuracy. |
| **MAE** | The average number of patients the model was off by per month. Treats all errors equally. |
| **RMSE** | Similar to MAE but squares errors first, so large spikes are penalised more heavily. |
| **MSE** | The raw squared error used internally during model training. |
        """)

    # Combine the history and the future into one dataset for plotting
    historical_plot = train_data.rename(columns={"ds": "Date", "y": "Value"}).copy()
    historical_plot["Series"] = "Historical"

    forecast_plot = forecast_df[["Date", "Predicted Attendances"]].rename(
        columns={"Predicted Attendances": "Value"}
    ).copy()
    forecast_plot["Series"] = "Forecast"

    combined_plot = pd.concat([historical_plot, forecast_plot])

    fc_col1, fc_col2 = st.columns([3, 1])

    with fc_col1.container(border=True):
        st.markdown(f"### Attendance Forecast — {model_type} ({horizon} months ahead)")

        forecast_chart = (
            alt.Chart(combined_plot)
            .mark_line(point=True)
            .encode(
                x=alt.X("Date:T", title=None),
                y=alt.Y("Value:Q", title="Total Attendances", axis=alt.Axis(format="~s")),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["Historical", "Forecast"],
                        range=[NHS_BLUE, NHS_RED]
                    ),
                    legend=alt.Legend(orient="bottom", title=None)
                ),
                # Make the forecast future line dashed
                strokeDash=alt.condition(
                    alt.datum.Series == "Forecast",
                    alt.value([6, 4]),
                    alt.value([0])
                ),
                tooltip=[
                    alt.Tooltip("Date:T", format="%b %Y"),
                    alt.Tooltip("Value:Q", format=",")
                ]
            )
            .properties(height=500)
            .interactive()
        )

        st.altair_chart(style_chart(forecast_chart), use_container_width=True, theme=None)

    # Show the raw numbers in a table next to the chart
    with fc_col2.container(border=True):
        st.markdown("### Forecasted Data")

        display_df = forecast_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%b %Y")
        display_df = display_df.rename(columns={
            "Predicted Attendances": "Attendances",
            "Predicted Admissions":  "Admissions"
        })

        st.dataframe(display_df.set_index("Date"), use_container_width=True, height=500)


# This is the main block that runs when you launch Streamlit
def main():
    setup_page()
    df = load_data()

    if df.empty:
        return

    st.markdown(
        """
        <div class="nh-header">
            <h1>National A&amp;E Analytics Dashboard</h1>
            <p>Monitor historical performance across NHS trusts and forecast future hospital demand Machine Learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Render the sidebar and get the user's choices back
    filtered_df, forecast_horizon, run_forecast = render_sidebar(df)
    
    # Render the top row of KPI numbers
    render_kpis(filtered_df)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Run the forecast if the button was clicked, otherwise show the regular charts
    if run_forecast:
        run_and_render_forecasts(filtered_df, forecast_horizon)
    else:
        render_historical_charts(filtered_df)

if __name__ == "__main__":
    main()
