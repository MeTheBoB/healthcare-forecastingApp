# My Final Year Project - NHS A&E Analytics Dashboard
# -------------------------------------------------------
# This is the main script for my interactive dashboard.
# I built it using a library called Streamlit which turns Python scripts
# into web apps without needing to know HTML or JavaScript properly.
#
# The dashboard does three main things:
#   1. Loads and cleans the NHS A&E data I collected
#   2. Shows historical charts so the user can explore patterns
#   3. Runs an AI forecast (Prophet or SARIMA) to predict future attendances
#
# To run this: open a terminal, cd to this folder, then type:
#   streamlit run streamlitApp.py

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from prophet import Prophet                              # Facebook's forecasting model
from sklearn.linear_model import LinearRegression        # Simple regression for admissions
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm                             # For SARIMA model
import warnings

# These libraries throw a lot of deprecation warnings that aren't relevant to my project,
# so I'm suppressing them to keep the terminal output readable
warnings.filterwarnings("ignore")


# -------------------------------------------------------
# COLOUR CONSTANTS
# -------------------------------------------------------
# I'm using the official NHS brand colours throughout so the dashboard
# looks like a real NHS product. Storing them here means if I ever need
# to change a colour I only have to do it in one place.

NHS_BLUE   = "#005EB8"   # main NHS blue
NHS_DARK   = "#003087"   # darker blue used for the sidebar and header
NHS_YELLOW = "#FFB81C"   # used for warnings and breach data
NHS_GREEN  = "#009639"   # used for targets that are being met
NHS_RED    = "#DA291C"   # used for the forecast line and failures
NHS_LIGHT  = "#E8EDEE"   # light grey used for dividers


# -------------------------------------------------------
# CUSTOM CSS STYLING
# -------------------------------------------------------
# Streamlit's default look is quite plain and purple, so I injected my own
# CSS to override it and make everything match the NHS visual style.
# The f-string below lets me reuse the colour constants I defined above
# directly inside the CSS, which avoids hardcoding hex codes everywhere.

CUSTOM_CSS = f"""
<style>

/* Import the fonts I chose - DM Sans for body text, DM Serif Display for headings */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* Set the main background to a light grey so the white cards stand out */
.stApp {{
    background: #F4F6F9;
    font-family: 'DM Sans', sans-serif;
}}

/* Make the page wider and add some breathing room at the top */
.block-container {{
    padding-top: 2rem !important;
    max-width: 1400px !important;
}}

/* The big blue banner at the top of the page */
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

/* The white card boxes around each chart and KPI */
[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background: #FFFFFF;
    border-radius: 12px !important;
    border: 1px solid #DDE3E9 !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}}

/* Give the KPI metric boxes a silver-grey background so they look
   like proper dashboard tiles rather than plain white cards.
   I'm targeting containers that hold a metric widget specifically. */
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMetricValue"]) > div {{
    background: #E8ECF0 !important;
    border: 1px solid #C8D0DA !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08) !important;
}}

/* KPI metric values - using a fluid clamp() so the number scales down
   automatically if the card gets narrow, keeping all digits visible */
[data-testid="stMetricValue"] > div {{
    color: {NHS_BLUE} !important;
    font-size: clamp(1.1rem, 1.6vw, 1.6rem) !important;
    font-weight: 700 !important;
    white-space: nowrap !important;
    overflow: visible !important;
    line-height: 1.2 !important;
}}

/* KPI label text - slightly bigger than before and still uppercase */
[data-testid="stMetricLabel"] > div,
[data-testid="stMetricLabel"] p {{
    color: #4A5568 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    white-space: normal !important;
    line-height: 1.3 !important;
}}

/* KPI delta text (the pass/fail indicator under the 4-hour metric) */
[data-testid="stMetricDelta"] {{
    font-size: 0.78rem !important;
}}

/* Make the KPI tile itself taller so the number has room to breathe */
[data-testid="stVerticalBlockBorderWrapper"]:has([data-testid="stMetricValue"]) > div {{
    padding: 18px 16px !important;
    min-height: 110px !important;
}}

/* Make the tab titles bigger and visible - by default Streamlit makes them quite small */
button[data-baseweb="tab"] p {{
    color: #000000 !important;
    font-weight: 700 !important;
    font-size: 1.3rem !important;
}}
/* Highlight the currently selected tab in NHS blue */
button[data-baseweb="tab"][aria-selected="true"] p {{
    color: {NHS_BLUE} !important;
}}

/* Chart section headings */
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

/* Sidebar background - dark NHS navy */
[data-testid="stSidebar"] {{
    background: {NHS_DARK} !important;
}}
[data-testid="stSidebar"] * {{
    color: #FFFFFF !important;
}}
/* Make the selected filter tags in the dropdowns show as NHS blue */
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {{
    background: {NHS_BLUE} !important;
}}

/* The main Run Forecast button */
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

/* The date range slider handle */
div[data-testid="stSlider"] div[role="slider"] {{
    background: {NHS_BLUE} !important;
}}

/* Info and success notification boxes */
.stInfo {{
    border-left: 4px solid {NHS_BLUE} !important;
    border-radius: 8px !important;
}}
.stSuccess {{
    border-left: 4px solid {NHS_GREEN} !important;
    border-radius: 8px !important;
}}

/* The thin horizontal line between sections */
.section-divider {{
    border: none;
    border-top: 1px solid {NHS_LIGHT};
    margin: 24px 0;
}}

/* Custom HTML progress bars I built for the regional performance tab.
   I couldn't find a Streamlit widget that did what I wanted so I made my own. */
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

</style>
"""


# -------------------------------------------------------
# FUNCTION: setup_page()
# -------------------------------------------------------
# This has to be the very first Streamlit call in the script,
# before any st.write() or st.markdown() calls, otherwise Streamlit
# throws an error. It sets the browser tab title and injects my CSS.

def setup_page():
    st.set_page_config(
        layout="wide",
        page_title="NHS A&E Analytics",
        initial_sidebar_state="expanded"
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -------------------------------------------------------
# FUNCTION: load_data()
# -------------------------------------------------------
# Reads the CSV, cleans it up, and returns a pandas dataframe.
#
# I added @st.cache_data above the function - this is a Streamlit decorator
# that means the CSV is only actually read once when the app first loads.
# After that, Streamlit reuses the cached result. Without this, the file
# would be re-read every single time the user moves a slider or changes a
# filter, which would make the app very slow.

@st.cache_data
def load_data():
    # I'm using a relative path so the script works on any computer,
    # not just my own laptop. The CSV just needs to be in the same folder.
    try:
        df = pd.read_csv(r"C:\Users\DELL\Desktop\Uni25-26\Project\work\DATA\processedData\finalData.csv")
    except FileNotFoundError:
        st.error("Could not find finalData.csv - make sure it is in the same folder as this script.")
        return pd.DataFrame()   # return an empty dataframe so the app doesn't crash

    # Remove duplicate rows for the same trust in the same month.
    # These occasionally appear in the NHS source data due to data collection issues.
    df = df.drop_duplicates(subset=["MonthYear", "Organization"])

    # The MonthYear column comes in as a string like "2022-01-01".
    # I need to convert it to a proper datetime object so I can filter and sort by date.
    df["MonthYear"] = pd.to_datetime(df["MonthYear"], errors="coerce")

    # Drop any rows where the date conversion failed (errors="coerce" turns bad dates into NaT)
    df = df.dropna(subset=["MonthYear"])

    # I'm only using data from January 2022 onwards.
    # The data before this point was heavily distorted by COVID lockdowns,
    # which would make the AI model learn the wrong seasonal patterns.
    df = df[df["MonthYear"] >= "2022-01-01"]

    # Sort by date and reset the row numbers so the index is clean
    df = df.sort_values("MonthYear").reset_index(drop=True)

    # These percentage columns are pre-calculated in my data processing script,
    # but I'm recalculating them here as a backup in case they get dropped
    if "Within_4h_%" not in df.columns:
        df["Within_4h_%"] = (df["Total Attendances < 4h"] / df["Total Attendances"] * 100).round(1)
    if "Over_4h_%" not in df.columns:
        df["Over_4h_%"] = 100 - df["Within_4h_%"]

    return df


# -------------------------------------------------------
# FUNCTION: style_chart(chart)
# -------------------------------------------------------
# All my Altair charts need the same background and axis colours applied.
# Rather than copy-pasting these four lines onto every single chart,
# I made this helper function that I can call once at the end of each chart.

def style_chart(chart):
    return (
        chart
        .configure(background="white")
        .configure_legend(labelColor="#333", titleColor="#333", labelFontSize=12)
        .configure_axis(labelColor="#555", titleColor="#555", gridColor="#EEF2F5", domainColor="#DDE3E9")
        .configure_view(strokeWidth=0)
    )


# -------------------------------------------------------
# FUNCTION: get_covid_bands(df)
# -------------------------------------------------------
# This draws shaded coloured rectangles behind my line charts to show
# which time periods were Pre-COVID, During COVID, and Post-COVID.
# I think this is important context for interpreting the data because
# the attendance patterns look very different in each period.
#
# It works by finding the earliest and latest date for each CovidPeriod
# label in the data, then drawing a rectangle between those two dates.

def get_covid_bands(df):
    # If the CovidPeriod column doesn't exist for some reason, just return
    # an empty layer so the rest of the chart still works fine
    if "CovidPeriod" not in df.columns:
        return alt.layer()

    # Find the start and end date for each COVID period label
    period_ranges = (
        df.groupby("CovidPeriod")["MonthYear"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"CovidPeriod": "period", "min": "start", "max": "end"})
    )

    # Each period gets a different pastel colour
    colour_map = {
        "Pre-COVID":    "#A8C5E0",   # light blue
        "During COVID": "#FFD580",   # light amber
        "Post-COVID":   "#A8D8B0"    # light green
    }

    # Loop through each period and build a rectangle layer for it
    bands = []
    for _, row in period_ranges.iterrows():
        colour = colour_map.get(row["period"], "#CCCCCC")

        # Altair needs the data in a dataframe, even for a single rectangle
        band_data = pd.DataFrame({
            "start":  [row["start"]],
            "end":    [row["end"]],
            "period": [row["period"]]
        })

        band = (
            alt.Chart(band_data)
            .mark_rect(opacity=0.12, color=colour)  # low opacity so it sits behind the line
            .encode(
                x=alt.X("start:T"),
                x2=alt.X2("end:T")
            )
        )
        bands.append(band)

    # Stack all the rectangles into one Altair layer and return it
    return alt.layer(*bands)


# -------------------------------------------------------
# FUNCTION: render_sidebar(df)
# -------------------------------------------------------
# This builds the dark blue sidebar panel on the left.
# It contains all the user controls: region/trust dropdowns,
# date slider, forecast horizon pills, and the Run/Clear buttons.
#
# I decided to put all the filters in the sidebar so the main area
# stays clean and is used purely for charts and KPIs.
#
# The function returns three things back to main():
#   - filtered_df: the data after applying the user's filter choices
#   - forecast_horizon: how many months ahead they want to forecast
#   - run_forecast: True/False whether the forecast should currently be shown

def render_sidebar(df):
    with st.sidebar:

        # Big title for the sidebar
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

        # Region multi-select dropdown
        # I prepend "All Regions" as an option so the user can reset easily
        region_options = ["All Regions"] + sorted(df["Region"].dropna().unique().tolist())
        selected_regions = st.multiselect(
            "Regions",
            options=region_options,
            default=["All Regions"],
            label_visibility="collapsed"
        )

        # Only filter if the user has chosen specific regions (not "All Regions")
        if "All Regions" not in selected_regions and len(selected_regions) > 0:
            df = df[df["Region"].isin(selected_regions)]

        # Trust dropdown - this automatically updates to only show trusts
        # from whichever regions the user selected above
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

        # Date range slider
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

        # Apply the date filter - .dt.date converts the datetime column to
        # plain date objects so it can be compared against the slider values
        filtered_df = df[
            (df["MonthYear"].dt.date >= start_date) &
            (df["MonthYear"].dt.date <= end_date)
        ]

        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:16px 0'>", unsafe_allow_html=True)

        # Forecast horizon selector - I used st.pills() for this because it's
        # quicker to click than a dropdown and shows all options at once
        st.markdown("**AI Forecast**")
        forecast_options = [3, 4, 5, 6, 9, 12, 24]
        forecast_horizon = st.pills(
            "Horizon (months)",
            options=forecast_options,
            default=6,
            label_visibility="collapsed"
        )

        # I'm using session_state to remember whether the forecast is showing.
        # Without this, clicking an expander would reset the page and the forecast
        # would disappear. session_state persists across reruns.
        if "show_forecast" not in st.session_state:
            st.session_state.show_forecast = False

        # Run and Clear buttons sit side by side
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Run ML Forecast", type="primary", use_container_width=True):
                st.session_state.show_forecast = True
        with btn_col2:
            if st.button("Clear Forecast", use_container_width=True):
                st.session_state.show_forecast = False

        run_forecast = st.session_state.show_forecast

        # Small status text at the bottom of the sidebar showing how much data is selected
        st.markdown("<hr style='border-color:rgba(255,255,255,0.15);margin:20px 0 8px'>", unsafe_allow_html=True)
        n_trusts = filtered_df["Organization"].nunique()
        n_months = filtered_df["MonthYear"].nunique()
        st.markdown(
            f"<div style='color:rgba(255,255,255,0.5);font-size:0.75rem'>"
            f"{n_trusts:,} trusts selected over {n_months} months</div>",
            unsafe_allow_html=True
        )

    return filtered_df, forecast_horizon, run_forecast


# -------------------------------------------------------
# FUNCTION: render_kpis(df)
# -------------------------------------------------------
# Shows the six headline numbers (KPIs) at the top of the main page.
# KPI = Key Performance Indicator - these give the user an instant
# summary before they dig into the detailed charts below.
#
# The six KPIs I chose are:
#   1. Total Attendances
#   2. Emergency Admissions
#   3. 4-Hour Target % (with pass/fail indicator)
#   4. >4h Decision to Admit (trolley waits)
#   5. >12h Decision to Admit (longer trolley waits)
#   6. Non-A&E Admissions (patients admitted via other routes)

def render_kpis(df):
    if df.empty:
        st.info("No data available for the selected filters.")
        return

    # --- Calculate all the KPI values from the filtered data ---

    total_attendances = int(df["Total Attendances"].sum())
    total_admissions  = int(df["Total Emergency Admissions"].sum())

    # Work out what % of patients were seen within 4 hours
    patients_within_4h = df["Total Attendances < 4h"].sum()
    if total_attendances > 0:
        four_hour_pct = patients_within_4h / total_attendances * 100
    else:
        four_hour_pct = 0

    # Trolley wait columns - I check they exist first because older data might not have them
    over_4h          = int(df[">4h decision to admit"].sum())  if ">4h decision to admit"  in df.columns else None
    over_12h         = int(df[">12h decision to admit"].sum()) if ">12h decision to admit" in df.columns else None
    other_admissions = int(df["Other Emergency admissions (i.e not via A&E)"].sum()) if "Other Emergency admissions (i.e not via A&E)" in df.columns else None

    # Use 6 columns if all the data is available, otherwise just 3
    num_cols = 6 if over_12h is not None else 3
    cols = st.columns(num_cols)

    with cols[0].container(border=True):
        st.metric(label="Total Attendances", value=f"{total_attendances:,}")

    with cols[1].container(border=True):
        st.metric(label="Emergency Admissions", value=f"{total_admissions:,}")

    with cols[2].container(border=True):
        # The 4-hour target was updated by NHS England in 2025/26 to 78%
        # (previously the target was 95% but no trust had met that for years)
        # I show a green tick if they're above 78%, red cross if below
        if four_hour_pct >= 78:
            delta_text   = "Above 78% target"
            delta_colour = "normal"    # Streamlit uses "normal" for green
        else:
            delta_text   = "Below 78% target"
            delta_colour = "inverse"   # Streamlit uses "inverse" for red

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


# -------------------------------------------------------
# FUNCTION: render_historical_charts(df)
# -------------------------------------------------------
# This is the biggest function in the script. It draws all the historical
# charts organised into 4 tabs. I chose tabs rather than just stacking
# everything vertically because it keeps the page manageable and lets the
# user focus on one thing at a time.
#
# Tab 1: Monthly attendances over time (line chart + department donut)
# Tab 2: 4-hour performance and trolley waits
# Tab 3: Regional comparison (bar chart + progress bars)
# Tab 4: How patients got admitted (A&E vs other routes)

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


    # =====================================================
    # TAB 1: ATTENDANCES
    # =====================================================
    with tab1:
        # Split the row into a wide left column and narrow right column
        col_left, col_right = st.columns([2, 1])

        with col_left.container(border=True):
            st.markdown("### Monthly A&E Attendances")

            num_trusts = df["Organization"].nunique()

            # Smart grouping logic: if the user has selected more than 5 trusts,
            # showing them all as separate lines makes the chart unreadable.
            # So I automatically combine them into a single aggregate line instead.
            if num_trusts > 5:
                st.info(
                    f"Smart Grouping Active: {num_trusts} trusts are selected. "
                    f"The chart has been combined into a single trendline to keep it readable."
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
                # I layer the COVID shading underneath the line using the + operator in Altair
                att_chart = style_chart((get_covid_bands(df) + line_chart).properties(height=500).interactive())

            elif num_trusts > 1:
                # Between 2 and 5 trusts: show separate coloured lines so the user can compare them
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
                # Just one trust selected (or all trusts aggregated into one line)
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
                st.caption("Pre-COVID | During COVID | Post-COVID  (shaded background bands)")

        # Donut chart breaking down attendances by department type
        with col_right.container(border=True):
            st.markdown("### Department Mix")

            # There are three types of A&E departments in the NHS:
            # Type 1 = major A&E units (the big emergency departments)
            # Type 2 = single specialty units (e.g. eye casualty)
            # Type 3 = minor injury units / urgent treatment centres
            type_data = pd.DataFrame({
                "Type":  ["Type 1 (Major)", "Type 2 (Single Spec)", "Type 3 (Minor/UTC)"],
                "Count": [df["Type 1"].sum(), df["Type 2"].sum(), df["Type 3"].sum()]
            })

            # Calculate the percentage share of each type for the tooltip
            total_type     = type_data["Count"].sum()
            type_data["Pct"] = type_data["Count"] / total_type if total_type > 0 else 0

            base = alt.Chart(type_data).encode(
                theta=alt.Theta("Count:Q", stack=True),
                color=alt.Color(
                    "Type:N",
                    scale=alt.Scale(
                        domain=type_data["Type"].tolist(),
                        range=[NHS_BLUE, NHS_YELLOW, NHS_GREEN]
                    ),
                    # columns=1 stacks the legend vertically so each label gets its own line
                    # labelLimit=0 means no truncation - the full text always shows
                    legend=alt.Legend(
                        orient="bottom",
                        title=None,
                        labelLimit=0,
                        columns=1,
                        labelFontSize=13,
                        symbolSize=120
                    )
                ),
                tooltip=[
                    alt.Tooltip("Type:N"),
                    alt.Tooltip("Pct:Q", format=".1%", title="%"),
                    alt.Tooltip("Count:Q", format=",.0f")
                ]
            )

            # innerRadius makes it a donut rather than a solid pie chart
            arc   = base.mark_arc(innerRadius=70)
            # Add the percentage label directly on each segment
            label = base.mark_text(radius=115, fontSize=14, fontWeight="bold", fill="white").encode(
                text=alt.Text("Pct:Q", format=".0%")
            )

            # Extra height to fit the stacked legend below the donut
            st.altair_chart(style_chart((arc + label).properties(height=560)), use_container_width=True, theme=None)


    # =====================================================
    # TAB 2: 4-HOUR PERFORMANCE
    # =====================================================
    with tab2:
        col_a, col_b = st.columns(2)

        # Stacked bar chart: blue = within 4h, yellow = breached 4h
        with col_a.container(border=True):
            st.markdown("### Attendances vs 4-Hour Target")

            # Group by month and sum the within/over 4h columns
            bar_data = df.groupby("MonthYear", as_index=False)[
                ["Total Attendances < 4h", "Total Attendances > 4 hours"]
            ].sum()

            # Rename for cleaner labels in the chart legend
            bar_data = bar_data.rename(columns={
                "Total Attendances < 4h":      "Within 4h",
                "Total Attendances > 4 hours": "Breached 4h"
            })

            # Altair needs data in "long" format (one row per category per month)
            # rather than "wide" format (one row per month with two columns).
            # The melt() function converts it from wide to long.
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

        # Line chart showing Type 1 performance against the 78% target
        with col_b.container(border=True):
            st.markdown("### Type 1 Performance Over Time")

            # I only show this chart if the column exists in the data
            if "Type1_Under4h_%" in df.columns:

                # Average the Type 1 performance across all selected trusts by month
                t1_data = df.groupby("MonthYear", as_index=False)["Type1_Under4h_%"].mean()

                # The actual performance line
                perf_line = (
                    alt.Chart(t1_data)
                    .mark_line(point=True, color=NHS_BLUE, strokeWidth=2.5)
                    .encode(
                        x=alt.X("MonthYear:T", title=None),
                        y=alt.Y(
                            "Type1_Under4h_%:Q",
                            title="% Within 4h",
                            scale=alt.Scale(domain=[60, 100])  # Fix axis from 60-100 not 0-100
                        ),
                        tooltip=[
                            alt.Tooltip("MonthYear:T", format="%b %Y"),
                            alt.Tooltip("Type1_Under4h_%:Q", format=".1f", title="% Within 4h")
                        ]
                    )
                )

                # A horizontal dashed line showing the 78% target
                # I build it from a tiny single-row dataframe because that's how Altair works
                target_line = (
                    alt.Chart(pd.DataFrame({"y": [78.0]}))
                    .mark_rule(color=NHS_YELLOW, strokeDash=[4, 3], strokeWidth=1.8)
                    .encode(y=alt.Y("y:Q"))
                )

                # A small text label at the right end of the target line
                target_label = (
                    alt.Chart(pd.DataFrame({
                        "x":    [t1_data["MonthYear"].max()],   # position at the last month
                        "y":    [78.7],                          # slightly above the line
                        "text": ["78% target (2025/26)"]
                    }))
                    .mark_text(align="right", fontSize=10, color="#B07800", fontWeight="bold")
                    .encode(
                        x=alt.X("x:T"),
                        y=alt.Y("y:Q"),
                        text="text:N"
                    )
                )

                # Layer all three elements together using + (Altair's layering operator)
                combined = (perf_line + target_line + target_label).properties(height=340).interactive()
                st.altair_chart(style_chart(combined), use_container_width=True, theme=None)
                st.caption("Dashed line = 78% NHS England target for 2025/26")

            else:
                st.info("Type 1 performance column not found in the data.")

        # Full-width area chart for trolley waits (below the two charts above)
        if ">4h decision to admit" in df.columns:
            st.markdown("")  # small spacer
            with st.container(border=True):
                st.markdown("### Trolley Waits - Decision to Admit")

                trolley_data = df.groupby("MonthYear", as_index=False)[
                    [">4h decision to admit", ">12h decision to admit"]
                ].sum()

                # Again, melt from wide to long format for Altair
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
                st.caption("Trolley waits measure how many patients waited over 4h or 12h for a bed after a decision to admit.")


    # =====================================================
    # TAB 3: REGIONAL BREAKDOWN
    # =====================================================
    with tab3:
        col_r1, col_r2 = st.columns(2)

        # Horizontal bar chart ranking regions by total attendances
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

        # Custom HTML progress bars showing each region's 4-hour performance
        # I built these in HTML/CSS because Streamlit doesn't have a native progress bar
        # that lets you set custom colours based on a value threshold
        with col_r2.container(border=True):
            st.markdown("### 4-Hour Performance by Region")

            # Calculate the % of patients seen within 4 hours for each region
            region_perf = (
                df.groupby("Region", as_index=False)
                .agg(
                    within_4h=("Total Attendances < 4h", "sum"),
                    total=("Total Attendances", "sum")
                )
            )
            region_perf["pct"] = region_perf["within_4h"] / region_perf["total"] * 100
            region_perf = region_perf.sort_values("pct", ascending=False)

            # Draw a progress bar for each region
            # Colour coding: green = above 78% target, yellow = 65-78%, red = below 65%
            for _, row in region_perf.iterrows():
                pct = row["pct"]

                if pct >= 78:
                    bar_colour = NHS_GREEN
                elif pct >= 65:
                    bar_colour = NHS_YELLOW
                else:
                    bar_colour = NHS_RED

                # The HTML/CSS for each individual bar
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


    # =====================================================
    # TAB 4: ADMISSION PATHWAYS
    # =====================================================
    with tab4:
        col_p1, col_p2 = st.columns([3, 2])

        # Stacked area chart comparing A&E vs non-A&E admission routes over time
        with col_p1.container(border=True):
            st.markdown("### Emergency Admissions - via A&E vs Other")

            if "Other Emergency admissions (i.e not via A&E)" in df.columns:

                # Sum up admissions by pathway for each month
                # groupby + agg with named outputs (via_ae=...) is cleaner than renaming after
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
                    .properties(height=500)
                    .interactive()
                )
                st.altair_chart(style_chart(pathway_chart), use_container_width=True, theme=None)

        # Donut chart showing the overall split between the two pathways
        with col_p2.container(border=True):
            st.markdown("### Admission Route Split")

            total_via_ae = float(df["Total Emergency Admissions via A&E"].sum())
            total_other  = float(df["Other Emergency admissions (i.e not via A&E)"].sum())
            grand_total  = total_via_ae + total_other

            if grand_total > 0:
                route_data = pd.DataFrame({
                    "Route":      ["Via A&E", "Other"],
                    "Admissions": [total_via_ae, total_other],
                    # Store as fraction (0-1) so Altair's ".0%" format displays correctly
                    "Pct":        [total_via_ae / grand_total, total_other / grand_total]
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

                arc2   = base2.mark_arc(innerRadius=70)
                label2 = base2.mark_text(radius=115, fontSize=14, fontWeight="bold", fill="white").encode(
                    text=alt.Text("Pct:Q", format=".0%")
                )

                st.altair_chart(
                    style_chart((arc2 + label2).properties(height=500)),
                    use_container_width=True,
                    theme=None
                )


# -------------------------------------------------------
# FUNCTION: run_and_render_forecasts(df, horizon)
# -------------------------------------------------------
# This is the most technically complex part of the project.
# It runs one of two time series models depending on the forecast length,
# then uses a second model (Linear Regression) to predict admissions
# from the forecasted attendance figures.
#
# Model selection logic (I read about this in the Prophet docs and a few papers):
#   <= 3 months ahead  →  SARIMA
#       SARIMA is better for short-term forecasts because it puts more weight
#       on recent data points, which matters when you're only looking a few months ahead.
#
#   >  3 months ahead  →  Prophet
#       Prophet handles long-term seasonality better and is more robust
#       when there are outliers or trend changes (like post-COVID recovery).

def run_and_render_forecasts(df, horizon):

    # Aggregate total attendances by month across all selected trusts.
    # Prophet requires the date column to be called 'ds' and the value column 'y' -
    # these are Prophet's naming conventions, not mine.
    train_data = (
        df.groupby("MonthYear", as_index=False)["Total Attendances"]
        .sum()
        .rename(columns={"MonthYear": "ds", "Total Attendances": "y"})
        .sort_values("ds")
    )

    # Safety check - we need at least 24 months of data to learn a full yearly seasonal cycle.
    # If the slider is set to a short range this will trigger.
    if len(train_data) < 24:
        st.info("Not enough historical data. The model needs at least 24 months to detect yearly seasonality patterns. Please expand the date slider range.")
        st.session_state.show_forecast = False  # reset button so user doesn't get stuck
        return

    model_type = "SARIMA" if horizon <= 3 else "Prophet"

    # Show a loading spinner - these models can take 10-30 seconds to train
    with st.spinner(f"Training {model_type} model - this may take a moment..."):

        # =================================================
        # SARIMA MODEL
        # =================================================
        # SARIMA stands for Seasonal AutoRegressive Integrated Moving Average.
        # The parameters I chose (1,2,1)(1,1,1,12) came from reading about
        # common configurations for monthly NHS data and testing a few options.
        # m=12 tells SARIMA that the seasonality repeats every 12 months (yearly).
        if model_type == "SARIMA":

            # Convert the dataframe into a proper time series with monthly frequency.
            # asfreq("MS") = Monthly Start frequency. ffill() fills any gaps in the data.
            ts = train_data.set_index("ds")["y"].asfreq("MS").ffill()

            sarima_model = sm.tsa.statespace.SARIMAX(
                ts,
                order=(1, 2, 1),             # (p, d, q) - AR order, differencing, MA order
                seasonal_order=(1, 1, 1, 12), # (P, D, Q, m) - seasonal version of above
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)               # disp=False stops it printing all the training output

            # Generate predictions for the next 'horizon' months
            forecast_result = sarima_model.get_forecast(steps=horizon)

            forecast_df = pd.DataFrame({
                "Date": forecast_result.predicted_mean.index,
                "Predicted Attendances": forecast_result.predicted_mean.values.round(0).astype(int)
            })

            # To calculate accuracy, compare the model's fitted values against actual values.
            # I skip the first 13 months because SARIMA takes time to warm up and stabilise.
            y_true = ts.values[13:]
            y_pred = sarima_model.fittedvalues.values[13:]

        # =================================================
        # PROPHET MODEL
        # =================================================
        # Prophet was developed by Facebook/Meta for business forecasting.
        # It's very good at picking up yearly trends and seasonal patterns,
        # and it handles the kind of sudden shifts we see in NHS data (like COVID) well.
        else:
            prophet_model = Prophet(
                interval_width=0.95,        # 95% confidence interval
                seasonality_mode="additive" # additive = seasons add to the trend (vs multiplicative)
            )
            # I add a monthly seasonality on top of the default yearly one
            # because A&E attendances have clear month-to-month patterns too (e.g. winter peaks)
            prophet_model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
            prophet_model.fit(train_data)

            # make_future_dataframe creates a dataframe of future dates to forecast into
            future_dates    = prophet_model.make_future_dataframe(periods=horizon, freq="MS")
            prophet_output  = prophet_model.predict(future_dates)

            # The output contains both historical fit and future forecast in one dataframe.
            # I filter to only keep the future rows (beyond the end of the training data).
            future_only = prophet_output[prophet_output["ds"] > train_data["ds"].max()]

            forecast_df = pd.DataFrame({
                "Date": future_only["ds"],
                "Predicted Attendances": future_only["yhat"].round(0).astype(int)   # yhat = predicted value
            })

            # For accuracy metrics, compare Prophet's fitted historical values against actuals
            in_sample = prophet_output[prophet_output["ds"] <= train_data["ds"].max()]
            y_true    = train_data["y"].values
            y_pred    = in_sample["yhat"].values

        # =================================================
        # STAGE 2: LINEAR REGRESSION CASCADE
        # =================================================
        # Now that I have forecasted attendance numbers, I use a simple linear
        # regression to predict how many of those attendances will result in
        # an emergency hospital admission.
        #
        # The idea is: admissions and attendances have a strong linear relationship
        # (more people coming = more admissions). So I can train a regression on the
        # historical data and then apply it to the forecasted attendances.

        # Prepare the historical relationship between attendances and admissions
        lr_data = df.groupby("MonthYear", as_index=False)[
            ["Total Attendances", "Total Emergency Admissions"]
        ].sum()

        # Train the regression: Admissions = m * Attendances + c
        lr_model = LinearRegression()
        lr_model.fit(lr_data[["Total Attendances"]], lr_data["Total Emergency Admissions"])

        # Apply the trained regression to the forecasted attendances to get predicted admissions.
        # I need to rename the column to match the training data column name first.
        forecast_df["Predicted Admissions"] = lr_model.predict(
            forecast_df[["Predicted Attendances"]].rename(
                columns={"Predicted Attendances": "Total Attendances"}
            )
        ).astype(int)

        # =================================================
        # ACCURACY METRICS
        # =================================================
        # These tell us how accurate the model is on the historical data it was trained on.
        # A lower number = better. MAPE is the easiest to explain to a non-technical audience.
        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mse  = mean_squared_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100   # express as a percentage

    st.success(f"Model trained: {model_type} + Linear Regression cascade")

    # Show the accuracy metric cards
    st.markdown("### Model Accuracy (Historical Fit)")
    m1, m2, m3, m4 = st.columns(4)

    with m1.container(border=True):
        st.metric(label="MAPE - % Error", value=f"{mape:.2f}%")
    with m2.container(border=True):
        st.metric(label="MAE - Avg Patient Error", value=f"{mae:,.0f}")
    with m3.container(border=True):
        st.metric(label="RMSE - Penalty Error", value=f"{rmse:,.0f}")
    with m4.container(border=True):
        st.metric(label="MSE - Squared Error", value=f"{mse:,.0f}")

    # Expandable section explaining what each metric actually means in plain English
    with st.expander("What do these metrics mean?"):
        st.markdown("""
| Metric | What it tells you |
|--------|------------------|
| **MAPE** | Average error as a percentage. The easiest metric to explain - e.g. "the model is typically 3% off". |
| **MAE** | The average number of patients the model was wrong by each month. Treats all errors equally. |
| **RMSE** | Like MAE but squares the errors first, so big occasional misses are penalised more than small consistent ones. |
| **MSE** | The raw squared error used internally. Hard to interpret directly but very sensitive to outliers. |
        """)

    # Build the forecast chart by combining historical and future data into one dataframe
    historical_plot = train_data.rename(columns={"ds": "Date", "y": "Value"}).copy()
    historical_plot["Series"] = "Historical"

    forecast_plot = forecast_df[["Date", "Predicted Attendances"]].rename(
        columns={"Predicted Attendances": "Value"}
    ).copy()
    forecast_plot["Series"] = "Forecast"

    # Stack them vertically - pd.concat joins two dataframes with the same columns
    combined_plot = pd.concat([historical_plot, forecast_plot])

    fc_col1, fc_col2 = st.columns([3, 1])

    with fc_col1.container(border=True):
        st.markdown(f"### Attendance Forecast - {model_type} ({horizon} months ahead)")

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
                        range=[NHS_BLUE, NHS_RED]  # blue for history, red for forecast
                    ),
                    legend=alt.Legend(orient="bottom", title=None)
                ),
                # Make the forecast portion dashed so it's visually distinct
                strokeDash=alt.condition(
                    alt.datum.Series == "Forecast",
                    alt.value([6, 4]),   # dashed
                    alt.value([0])       # solid
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

    # Table showing the actual forecasted numbers next to the chart
    with fc_col2.container(border=True):
        st.markdown("### Forecasted Data")

        display_df = forecast_df.copy()
        display_df["Date"] = display_df["Date"].dt.strftime("%b %Y")  # format as "Jan 2025"
        display_df = display_df.rename(columns={
            "Predicted Attendances": "Attendances",
            "Predicted Admissions":  "Admissions"
        })

        st.dataframe(display_df.set_index("Date"), use_container_width=True, height=500)


# -------------------------------------------------------
# MAIN FUNCTION
# -------------------------------------------------------
# This is the entry point - Streamlit runs this function when you
# launch the app. It calls all the other functions in order.
#
# Order of execution:
#   1. setup_page()              - configure page settings and inject CSS
#   2. load_data()               - read and clean the CSV
#   3. header banner             - render the blue title banner
#   4. render_sidebar()          - build the filters and get the user's choices back
#   5. render_kpis()             - show the six headline numbers
#   6. charts OR forecast        - show whichever the user has asked for

def main():
    setup_page()
    df = load_data()

    # Stop here if the data file wasn't found
    if df.empty:
        return

    # Render the blue header banner at the top of the page
    st.markdown(
        """
        <div class="nh-header">
            <h1>National A&amp;E Analytics Dashboard</h1>
            <p>Monitor historical performance across NHS trusts and forecast future hospital demand using Machine Learning.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Build the sidebar and get the filtered data + forecast settings back
    filtered_df, forecast_horizon, run_forecast = render_sidebar(df)

    # Show the KPI summary cards
    render_kpis(filtered_df)

    # Thin divider line between KPIs and charts
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # Show either the forecast or the historical charts depending on the button state
    if run_forecast:
        run_and_render_forecasts(filtered_df, forecast_horizon)
    else:
        render_historical_charts(filtered_df)


# Standard Python convention - only run main() if this file is executed directly,
# not if it's imported as a module by another script
if __name__ == "__main__":
    main()