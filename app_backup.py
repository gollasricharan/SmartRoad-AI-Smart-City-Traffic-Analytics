
from pathlib import Path

import streamlit as st
import pandas as pd
import folium
import plotly.express as px

from streamlit_folium import st_folium
from folium.plugins import Fullscreen, Geocoder, MeasureControl


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartRoad AI",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

ACCIDENT_FILE = DATA_DIR / "raw" / "hyderabad_accidents.csv"
WEATHER_FILE = DATA_DIR / "raw" / "hyderabad_weather.csv"

PROCESSED_DIR = DATA_DIR / "processed"

MAP_HEIGHT = 680


# ============================================================
# 2. CITY CONFIGURATION
# ============================================================

CITIES = {
    "Hyderabad": {
        "state": "Telangana",
        "latitude": 17.3850,
        "longitude": 78.4867
    },
    "Bengaluru": {
        "state": "Karnataka",
        "latitude": 12.9716,
        "longitude": 77.5946
    },
    "Mumbai": {
        "state": "Maharashtra",
        "latitude": 19.0760,
        "longitude": 72.8777
    },
    "Delhi": {
        "state": "Delhi",
        "latitude": 28.6139,
        "longitude": 77.2090
    },
    "Chennai": {
        "state": "Tamil Nadu",
        "latitude": 13.0827,
        "longitude": 80.2707
    },
    "Pune": {
        "state": "Maharashtra",
        "latitude": 18.5204,
        "longitude": 73.8567
    }
}


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main {
        background-color: #0b1020;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .main-title {
        font-size: 44px;
        font-weight: 800;
        color: #f1f5f9;
    }

    .sub-title {
        color: #9ca3af;
        font-size: 15px;
    }

    .section-title {
        color: #f1f5f9;
        font-size: 26px;
        font-weight: 700;
    }

    div[data-testid="stMetric"] {
        background-color: #19263b;
        border: 1px solid #293951;
        padding: 15px;
        border-radius: 12px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 4. ROAD CATEGORY COLORS
# ============================================================

ROAD_COLORS = {
    "motorway": "#ff1744",
    "motorway_link": "#ff5252",
    "trunk": "#ff5722",
    "trunk_link": "#ff8a50",
    "primary": "#ff9800",
    "primary_link": "#ffb74d",
    "secondary": "#ffeb3b",
    "secondary_link": "#fff176",
    "tertiary": "#00c853",
    "tertiary_link": "#69f0ae",
    "residential": "#29b6f6",
    "living_street": "#00bcd4",
    "service": "#ab47bc",
    "unclassified": "#90a4ae",
    "footway": "#b0bec5",
    "path": "#78909c",
    "pedestrian": "#ce93d8",
    "cycleway": "#76ff03",
    "track": "#8d6e63",
    "steps": "#eeeeee",
    "construction": "#ef5350",
    "unknown": "#ffffff"
}


# ============================================================
# 5. INFRASTRUCTURE SCREENING WEIGHTS
# Illustrative weights, NOT accident-risk probabilities.
# ============================================================

ROAD_PRIORITY_WEIGHTS = {
    "motorway": 100,
    "motorway_link": 90,
    "trunk": 95,
    "trunk_link": 85,
    "primary": 90,
    "primary_link": 80,
    "secondary": 75,
    "secondary_link": 70,
    "tertiary": 60,
    "tertiary_link": 55,
    "residential": 40,
    "living_street": 35,
    "unclassified": 35,
    "service": 25,
    "services": 25,
    "track": 20,
    "cycleway": 30,
    "footway": 20,
    "path": 15,
    "pedestrian": 20,
    "steps": 10,
    "construction": 30,
    "elevator": 10,
    "unknown": 10
}


# ============================================================
# 6. FILE PATH HELPERS
# ============================================================

def get_city_folder(city):
    return PROCESSED_DIR / city.lower()


def get_city_csv(city):
    folder = get_city_folder(city)
    return folder / f"{city.lower()}_roads.csv"


def get_city_summary_file(city):
    folder = get_city_folder(city)
    return folder / "road_type_summary.csv"


def get_city_geojson(city):
    folder = get_city_folder(city)
    return folder / f"{city.lower()}_roads.geojson"


# ============================================================
# 7. DATA LOADING FUNCTIONS
# ============================================================

@st.cache_data(show_spinner=False)
def load_accidents():

    if not ACCIDENT_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(ACCIDENT_FILE)

        for column in ["year", "accidents"]:
            if column in df.columns:
                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        required = ["year", "accidents"]

        if not all(column in df.columns for column in required):
            return pd.DataFrame()

        df = df.dropna(subset=required)
        df = df.sort_values("year")

        return df

    except Exception as error:
        st.error(f"Could not load accident data: {error}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_weather():

    if not WEATHER_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(WEATHER_FILE)

        if "time" in df.columns:
            df["time"] = pd.to_datetime(
                df["time"],
                errors="coerce"
            )

        return df

    except Exception as error:
        st.error(f"Could not load weather data: {error}")
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_road_summary(city):

    summary_file = get_city_summary_file(city)

    if not summary_file.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(summary_file)

        required = [
            "road_type",
            "road_segments",
            "total_length_km"
        ]

        if not all(column in df.columns for column in required):
            return pd.DataFrame()

        df["road_segments"] = pd.to_numeric(
            df["road_segments"],
            errors="coerce"
        ).fillna(0)

        df["total_length_km"] = pd.to_numeric(
            df["total_length_km"],
            errors="coerce"
        ).fillna(0)

        df["road_type"] = (
            df["road_type"]
            .fillna("unknown")
            .astype(str)
        )

        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def get_available_road_cities():

    available = []

    for city in CITIES:

        csv_file = get_city_csv(city)
        summary_file = get_city_summary_file(city)

        if csv_file.exists() and summary_file.exists():
            available.append(city)

    return available


@st.cache_data(show_spinner=False)
def get_city_road_statistics(city):

    summary = load_road_summary(city)

    if summary.empty:
        return {
            "segments": 0,
            "length": 0.0,
            "categories": 0
        }

    return {
        "segments": int(summary["road_segments"].sum()),
        "length": float(summary["total_length_km"].sum()),
        "categories": int(summary["road_type"].nunique())
    }


@st.cache_data(show_spinner=False)
def load_road_csv_for_export(city):

    csv_file = get_city_csv(city)

    if not csv_file.exists():
        return pd.DataFrame()

    # This potentially large file is read only when requested.
    return pd.read_csv(csv_file)


# ============================================================
# 8. LOAD COMMON DATA
# ============================================================

accidents = load_accidents()
weather = load_weather()

available_road_cities = get_available_road_cities()


# ============================================================
# 9. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚦 SMARTROAD AI")
    st.caption("ROAD SAFETY INTELLIGENCE")

    st.divider()

    st.markdown("### NAVIGATION")

    page = st.radio(
        "Choose Dashboard Page",
        [
            "Overview",
            "Interactive Map",
            "Accident Analysis",
            "Road Risk Intelligence",
            "Traffic & Weather",
            "Reports & Insights"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown("### 📍 Select City")

    selected_city = st.selectbox(
        "City",
        list(CITIES.keys())
    )

    city_info = CITIES[selected_city]

    latitude = city_info["latitude"]
    longitude = city_info["longitude"]

    st.caption(city_info["state"])

    st.divider()

    st.success("Dashboard online")

    if selected_city in available_road_cities:
        st.success(f"{selected_city} road data available")
    else:
        st.warning("Road summary data not available")

    if selected_city == "Hyderabad":
        st.info("Accident and weather datasets available.")
    else:
        st.caption(
            "Accident and weather datasets are currently "
            "available only for Hyderabad."
        )


# ============================================================
# 10. LOAD LIGHTWEIGHT CITY DATA
# ============================================================

road_summary = load_road_summary(selected_city)

road_stats = get_city_road_statistics(selected_city)


# ============================================================
# 11. COMMON HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 SmartRoad AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'India Road Accident Risk & Traffic Intelligence System'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 12. OVERVIEW PAGE
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">'
        'City Intelligence Overview'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(f"Selected city: **{selected_city}**")

    road_count = road_stats["segments"]
    road_length = road_stats["length"]
    road_categories = road_stats["categories"]

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Road Segments",
        f"{road_count:,}"
    )

    c2.metric(
        "Mapped Length",
        f"{road_length:,.1f} km"
    )

    c3.metric(
        "Road Categories",
        road_categories
    )

    c4.metric(
        "Cities With Road Data",
        len(available_road_cities)
    )

    st.divider()

    st.markdown("## Road Network Coverage")

    if not road_summary.empty:

        st.success(
            f"Road summary available for {selected_city}."
        )

        st.markdown("### Road Type Summary")

        display_summary = road_summary.rename(
            columns={
                "road_type": "Road Category",
                "road_segments": "Road Segments",
                "total_length_km": "Total Length (km)"
            }
        )

        display_summary["Total Length (km)"] = (
            display_summary["Total Length (km)"].round(2)
        )

        st.dataframe(
            display_summary,
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "Statistics are calculated from the existing road summary "
            "CSV. Mapped length can include service roads, footways, "
            "paths and other highway-tagged features. It is not "
            "exclusively motor-vehicle road mileage."
        )

    else:

        st.warning(
            f"Road summary is not available for {selected_city}."
        )

    st.divider()

    st.markdown("## Accident Overview")

    if selected_city != "Hyderabad":

        st.info(
            "Historical accident data is currently available "
            "only for Hyderabad. Other cities will be added "
            "when verified datasets are available."
        )

    elif not accidents.empty:

        overview_data = accidents.copy()

        overview_data["Year"] = (
            overview_data["year"].astype(int).astype(str)
        )

        overview_figure = px.line(
            overview_data,
            x="Year",
            y="accidents",
            markers=True,
            title="Annual Accident Trend",
            labels={
                "Year": "Year",
                "accidents": "Number of Accidents"
            }
        )

        overview_figure.update_layout(
            height=350,
            xaxis=dict(type="category"),
            margin=dict(l=10, r=10, t=50, b=10)
        )

        st.plotly_chart(
            overview_figure,
            use_container_width=True
        )

        st.caption(
            "Annual city-level accident counts for Hyderabad."
        )

    else:

        st.warning("Hyderabad accident dataset not found.")


# ============================================================
# 13. INTERACTIVE MAP PAGE
# ============================================================

elif page == "Interactive Map":

    st.markdown(
        '<div class="section-title">'
        '🗺️ Interactive Road Network Map'
        '</div>',
        unsafe_allow_html=True
    )

    if selected_city not in available_road_cities:

        st.warning(
            f"Road summary data is not available for {selected_city}."
        )

    else:

        st.markdown(
            f"### {selected_city}, {city_info['state']}"
        )

        st.caption(
            "The map uses OpenStreetMap map tiles. These tiles display "
            "the road network without loading the large local GeoJSON "
            "files into your computer's memory."
        )

        col1, col2 = st.columns(2)

        with col1:

            map_style = st.selectbox(
                "Map style",
                [
                    "Street Map",
                    "Dark Map",
                    "Satellite"
                ]
            )

        with col2:

            zoom_level = st.slider(
                "Map zoom level",
                min_value=9,
                max_value=17,
                value=12
            )

        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=zoom_level,
            tiles=None,
            control_scale=True
        )

        if map_style == "Street Map":

            folium.TileLayer(
                "OpenStreetMap",
                name="Street Map"
            ).add_to(m)

        elif map_style == "Dark Map":

            folium.TileLayer(
                "CartoDB dark_matter",
                name="Dark Map"
            ).add_to(m)

        else:

            folium.TileLayer(
                tiles=(
                    "https://server.arcgisonline.com/"
                    "ArcGIS/rest/services/World_Imagery/"
                    "MapServer/tile/{z}/{y}/{x}"
                ),
                attr="Esri",
                name="Satellite"
            ).add_to(m)

        folium.Marker(
            [latitude, longitude],
            popup=f"{selected_city} City Centre",
            tooltip=selected_city,
            icon=folium.Icon(
                color="red",
                icon="info-sign"
            )
        ).add_to(m)

        Fullscreen(
            position="topright"
        ).add_to(m)

        Geocoder(
            collapsed=True,
            position="topleft"
        ).add_to(m)

        MeasureControl(
            position="bottomleft"
        ).add_to(m)

        folium.LayerControl(
            collapsed=True
        ).add_to(m)

        st_folium(
            m,
            width=None,
            height=MAP_HEIGHT,
            key=f"map_{selected_city}_{map_style}"
        )

        st.divider()

        st.markdown("### Road Network Statistics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Mapped Road Segments",
            f"{road_stats['segments']:,}"
        )

        c2.metric(
            "Mapped Length",
            f"{road_stats['length']:,.1f} km"
        )

        c3.metric(
            "Road Categories",
            road_stats["categories"]
        )

        if not road_summary.empty:

            st.markdown("### Available Road Categories")

            st.dataframe(
                road_summary.rename(
                    columns={
                        "road_type": "Road Category",
                        "road_segments": "Road Segments",
                        "total_length_km": "Total Length (km)"
                    }
                ),
                use_container_width=True,
                hide_index=True
            )

        st.caption(
            "Source: OpenStreetMap. Map tiles show the mapped street "
            "network; the local GeoJSON road geometries are not loaded "
            "as individual interactive overlays in this optimized map."
        )


# ============================================================
# 14. ACCIDENT ANALYSIS DASHBOARD
# ============================================================

elif page == "Accident Analysis":

    st.markdown(
        '<div class="section-title">'
        '📊 Road Accident Analysis Dashboard'
        '</div>',
        unsafe_allow_html=True
    )

    if selected_city != "Hyderabad":

        st.warning(
            "Accident records currently available in this project "
            "are for Hyderabad only."
        )

    elif accidents.empty:

        st.error("Accident dataset not found.")

    else:

        st.markdown(
            "### Accident Statistics and Year-wise Trends"
        )

        available_years = sorted(
            accidents["year"].astype(int).unique().tolist()
        )

        selected_years = st.select_slider(
            "Select analysis period",
            options=available_years,
            value=(
                available_years[0],
                available_years[-1]
            )
        )

        filtered_accidents = accidents[
            (accidents["year"] >= selected_years[0]) &
            (accidents["year"] <= selected_years[1])
        ].copy()

        if filtered_accidents.empty:

            st.warning(
                "No accident records in this period."
            )

        else:

            total_accidents = filtered_accidents[
                "accidents"
            ].sum()

            average_accidents = filtered_accidents[
                "accidents"
            ].mean()

            highest_row = filtered_accidents.loc[
                filtered_accidents["accidents"].idxmax()
            ]

            lowest_row = filtered_accidents.loc[
                filtered_accidents["accidents"].idxmin()
            ]

            st.markdown("### Key Performance Indicators")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Total Accidents",
                f"{total_accidents:,.0f}"
            )

            c2.metric(
                "Yearly Average",
                f"{average_accidents:,.0f}"
            )

            c3.metric(
                "Highest Year",
                f"{int(highest_row['year'])}",
                help=f"{highest_row['accidents']:,.0f} accidents"
            )

            c4.metric(
                "Lowest Year",
                f"{int(lowest_row['year'])}",
                help=f"{lowest_row['accidents']:,.0f} accidents"
            )

            st.divider()

            st.markdown("### 📈 Year-wise Accident Trend")

            chart_data = filtered_accidents.copy()

            chart_data["Year"] = (
                chart_data["year"].astype(int).astype(str)
            )

            trend_figure = px.line(
                chart_data,
                x="Year",
                y="accidents",
                markers=True,
                labels={
                    "Year": "Year",
                    "accidents": "Number of Accidents"
                }
            )

            trend_figure.update_layout(
                height=350,
                xaxis=dict(type="category"),
                margin=dict(l=10, r=10, t=20, b=10)
            )

            st.plotly_chart(
                trend_figure,
                use_container_width=True
            )

            st.markdown("### 📊 Annual Accident Comparison")

            bar_figure = px.bar(
                chart_data,
                x="Year",
                y="accidents",
                labels={
                    "Year": "Year",
                    "accidents": "Number of Accidents"
                }
            )

            bar_figure.update_layout(
                height=350,
                xaxis=dict(type="category"),
                margin=dict(l=10, r=10, t=20, b=10)
            )

            st.plotly_chart(
                bar_figure,
                use_container_width=True
            )

            st.divider()

            st.markdown("### Year-on-Year Change")

            comparison = filtered_accidents.copy()

            comparison["change"] = (
                comparison["accidents"].diff()
            )

            comparison["change_percent"] = (
                comparison["accidents"].pct_change() * 100
            )

            comparison["change_percent"] = (
                comparison["change_percent"].round(2)
            )

            comparison["change"] = (
                comparison["change"].round(0)
            )

            comparison_display = comparison.rename(
                columns={
                    "year": "Year",
                    "city": "City",
                    "accidents": "Accidents",
                    "change": "Change in Accidents",
                    "change_percent": "Change (%)"
                }
            )

            st.dataframe(
                comparison_display,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.markdown("### 💡 Automatic Insights")

            first_row = filtered_accidents.iloc[0]
            last_row = filtered_accidents.iloc[-1]

            if len(filtered_accidents) >= 2:

                first_value = first_row["accidents"]
                last_value = last_row["accidents"]

                if first_value != 0:

                    percentage_change = (
                        (last_value - first_value) /
                        first_value
                    ) * 100

                    if percentage_change > 0:

                        st.warning(
                            f"Accidents increased by "
                            f"{percentage_change:.2f}% between "
                            f"{int(first_row['year'])} and "
                            f"{int(last_row['year'])}."
                        )

                    elif percentage_change < 0:

                        st.success(
                            f"Accidents decreased by "
                            f"{abs(percentage_change):.2f}% between "
                            f"{int(first_row['year'])} and "
                            f"{int(last_row['year'])}."
                        )

                    else:

                        st.info(
                            "Accident counts remained unchanged "
                            "between the selected period's first "
                            "and last years."
                        )

            st.info(
                f"The highest recorded annual accident count in "
                f"the selected period was "
                f"{highest_row['accidents']:,.0f} "
                f"in {int(highest_row['year'])}."
            )

            st.info(
                f"The lowest recorded annual accident count in "
                f"the selected period was "
                f"{lowest_row['accidents']:,.0f} "
                f"in {int(lowest_row['year'])}."
            )

            st.divider()

            st.markdown("### 📋 Accident Dataset")

            st.dataframe(
                filtered_accidents,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                label="⬇️ Download Filtered Accident Data",
                data=filtered_accidents.to_csv(index=False),
                file_name="hyderabad_accident_analysis.csv",
                mime="text/csv"
            )

            st.caption(
                "Analysis is based on annual city-level accident "
                "counts. Percentage changes describe recorded counts "
                "and do not establish causes or individual accident locations."
            )


# ============================================================
# 15. ROAD RISK INTELLIGENCE
# ============================================================

elif page == "Road Risk Intelligence":

    st.markdown(
        '<div class="section-title">'
        '⚠️ Road Risk Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Infrastructure-based screening of the mapped road network."
    )

    if road_summary.empty:

        st.warning(
            f"Road summary data is not available for {selected_city}."
        )

    else:

        risk_data = road_summary.copy()

        risk_data["screening_weight"] = (
            risk_data["road_type"]
            .map(ROAD_PRIORITY_WEIGHTS)
            .fillna(10)
        )

        total_segments = risk_data["road_segments"].sum()
        total_length = risk_data["total_length_km"].sum()

        if total_segments > 0:

            risk_data["segment_share_percent"] = (
                risk_data["road_segments"] /
                total_segments * 100
            )

        else:

            risk_data["segment_share_percent"] = 0

        if total_length > 0:

            risk_data["length_share_percent"] = (
                risk_data["total_length_km"] /
                total_length * 100
            )

        else:

            risk_data["length_share_percent"] = 0

        risk_data["segment_share_percent"] = (
            risk_data["segment_share_percent"].round(2)
        )

        risk_data["length_share_percent"] = (
            risk_data["length_share_percent"].round(2)
        )

        risk_data = risk_data.sort_values(
            "screening_weight",
            ascending=False
        )

        st.markdown("### Infrastructure Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Mapped Road Segments",
            f"{total_segments:,.0f}"
        )

        c2.metric(
            "Mapped Length",
            f"{total_length:,.1f} km"
        )

        c3.metric(
            "Road Categories",
            risk_data["road_type"].nunique()
        )

        priority_categories = [
            "motorway",
            "motorway_link",
            "trunk",
            "trunk_link",
            "primary",
            "primary_link"
        ]

        available_priority_categories = (
            set(risk_data["road_type"].unique())
            .intersection(priority_categories)
        )

        c4.metric(
            "Priority Road Classes",
            len(available_priority_categories)
        )

        st.divider()

        st.markdown("### Infrastructure Priority Screening")

        st.info(
            "This section ranks road categories using an illustrative "
            "road-hierarchy weight. It helps identify infrastructure "
            "classes for further investigation. It is NOT a measured "
            "accident-risk score or a prediction of crash probability."
        )

        selected_risk_types = st.multiselect(
            "Select road categories for analysis",
            options=sorted(risk_data["road_type"].unique()),
            default=sorted(risk_data["road_type"].unique()),
            key=f"risk_category_filter_{selected_city}"
        )

        filtered_risk = risk_data[
            risk_data["road_type"].isin(selected_risk_types)
        ].copy()

        if filtered_risk.empty:

            st.warning(
                "Select at least one road category."
            )

        else:

            st.markdown("### 1. Road Category Distribution")

            category_counts = (
                filtered_risk
                .set_index("road_type")[["road_segments"]]
                .sort_values("road_segments", ascending=False)
            )

            st.bar_chart(category_counts)

            st.markdown("### 2. Road Length by Category")

            length_chart = (
                filtered_risk
                .set_index("road_type")[["total_length_km"]]
                .sort_values("total_length_km", ascending=False)
            )

            st.bar_chart(length_chart)

            st.markdown("### 3. Infrastructure Screening Weights")

            weight_chart = (
                filtered_risk
                .set_index("road_type")[["screening_weight"]]
                .sort_values(
                    "screening_weight",
                    ascending=False
                )
            )

            st.bar_chart(weight_chart)

            st.markdown("### 4. Road Category Screening Table")

            display_summary = filtered_risk.rename(
                columns={
                    "road_type": "Road Category",
                    "road_segments": "Road Segments",
                    "total_length_km": "Total Length (km)",
                    "screening_weight": "Screening Weight",
                    "segment_share_percent": "Segment Share (%)",
                    "length_share_percent": "Length Share (%)"
                }
            )

            display_summary["Total Length (km)"] = (
                display_summary["Total Length (km)"].round(2)
            )

            st.dataframe(
                display_summary,
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### 5. Categories for Further Investigation"
            )

            top_categories = filtered_risk.head(5)

            for _, row in top_categories.iterrows():

                st.markdown(
                    f"**{row['road_type'].replace('_', ' ').title()}**"
                    f" — {int(row['road_segments']):,} segments, "
                    f"{row['total_length_km']:.2f} km mapped length, "
                    f"screening weight "
                    f"{row['screening_weight']:.1f}/100"
                )

            st.download_button(
                label="⬇️ Download Infrastructure Screening Report",
                data=filtered_risk.to_csv(index=False),
                file_name=(
                    f"{selected_city.lower()}_"
                    "infrastructure_screening.csv"
                ),
                mime="text/csv"
            )

        st.divider()

        st.markdown("### How to Interpret This Analysis")

        st.markdown("""
        - **Road segments:** Number of mapped OpenStreetMap road features.
        - **Mapped length:** Sum of mapped road geometry lengths.
        - **Screening weight:** Illustrative weight assigned to a road category.
        - **Segment share:** Percentage of mapped features in a category.
        - **Length share:** Percentage of mapped geometry length in a category.

        Higher screening weights indicate road classes that may deserve
        infrastructure review. They do not mean that those roads have
        more accidents.
        """)

        st.warning(
            "Actual road-specific accident risk requires verified accident "
            "locations, dates, severity, traffic exposure and road conditions. "
            "These are not available in the current dataset."
        )


# ============================================================
# 16. TRAFFIC AND WEATHER
# ============================================================

elif page == "Traffic & Weather":

    st.markdown(
        '<div class="section-title">'
        '🌦️ Traffic & Weather Intelligence'
        '</div>',
        unsafe_allow_html=True
    )

    st.info(
        "Historical weather information is available. "
        "Live traffic data is not currently connected."
    )

    if selected_city != "Hyderabad":

        st.warning(
            "Historical weather data is currently available "
            "only for Hyderabad."
        )

    elif weather.empty:

        st.warning("Weather dataset not found.")

    elif "time" not in weather.columns:

        st.error(
            "The weather dataset does not contain a time column."
        )

    else:

        weather_data = weather.copy()

        weather_data["time"] = pd.to_datetime(
            weather_data["time"],
            errors="coerce"
        )

        weather_data = weather_data.dropna(
            subset=["time"]
        )

        numeric_columns = [
            column for column in [
                "temperature_2m",
                "relative_humidity_2m",
                "precipitation",
                "rain"
            ]
            if column in weather_data.columns
        ]

        if weather_data.empty or not numeric_columns:

            st.warning(
                "No valid weather records or numeric weather columns "
                "were found."
            )

        else:

            for column in numeric_columns:

                weather_data[column] = pd.to_numeric(
                    weather_data[column],
                    errors="coerce"
                )

            weather_data["year"] = (
                weather_data["time"].dt.year
            )

            available_years = sorted(
                weather_data["year"]
                .dropna()
                .astype(int)
                .unique()
                .tolist()
            )

            if not available_years:

                st.warning(
                    "No valid years were found in the weather dataset."
                )

            else:

                variable_labels = {
                    "temperature_2m": "Temperature",
                    "relative_humidity_2m": "Relative Humidity",
                    "precipitation": "Precipitation",
                    "rain": "Rain"
                }

                st.markdown("### 📅 Weather Analysis Filters")

                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:

                    selected_year = st.selectbox(
                        "Select year",
                        options=available_years,
                        index=(
                            available_years.index(2023)
                            if 2023 in available_years
                            else len(available_years) - 1
                        )
                    )

                with filter_col2:

                    selected_variable = st.selectbox(
                        "Select weather parameter",
                        options=numeric_columns,
                        format_func=lambda value: variable_labels.get(
                            value,
                            value.replace("_", " ").title()
                        ),
                        index=(
                            numeric_columns.index("temperature_2m")
                            if "temperature_2m" in numeric_columns
                            else 0
                        )
                    )

                year_df = weather_data[
                    weather_data["year"] == selected_year
                ].copy()

                st.divider()

                st.markdown("### Weather Performance Indicators")

                metric_cols = st.columns(4)

                if "temperature_2m" in numeric_columns:

                    avg_temp = year_df[
                        "temperature_2m"
                    ].mean()

                    metric_cols[0].metric(
                        "Average Temperature",
                        f"{avg_temp:.1f} °C"
                        if pd.notna(avg_temp)
                        else "N/A"
                    )

                else:

                    metric_cols[0].metric(
                        "Average Temperature",
                        "N/A"
                    )

                if "relative_humidity_2m" in numeric_columns:

                    avg_humidity = year_df[
                        "relative_humidity_2m"
                    ].mean()

                    metric_cols[1].metric(
                        "Average Humidity",
                        f"{avg_humidity:.1f}%"
                        if pd.notna(avg_humidity)
                        else "N/A"
                    )

                else:

                    metric_cols[1].metric(
                        "Average Humidity",
                        "N/A"
                    )

                if "precipitation" in numeric_columns:

                    total_precipitation = year_df[
                        "precipitation"
                    ].sum()

                    metric_cols[2].metric(
                        "Total Precipitation",
                        f"{total_precipitation:,.1f} mm"
                    )

                else:

                    metric_cols[2].metric(
                        "Total Precipitation",
                        "N/A"
                    )

                if "rain" in numeric_columns:

                    rainy_hours = int(
                        (year_df["rain"].fillna(0) > 0).sum()
                    )

                    metric_cols[3].metric(
                        "Rainy Hours",
                        f"{rainy_hours:,}"
                    )

                else:

                    metric_cols[3].metric(
                        "Rainy Hours",
                        "N/A"
                    )

                month_names = [
                    "January", "February", "March",
                    "April", "May", "June",
                    "July", "August", "September",
                    "October", "November", "December"
                ]

                month_short_names = [
                    "Jan", "Feb", "Mar", "Apr",
                    "May", "Jun", "Jul", "Aug",
                    "Sep", "Oct", "Nov", "Dec"
                ]

                year_df["month"] = (
                    year_df["time"].dt.month
                )

                selected_label = variable_labels.get(
                    selected_variable,
                    selected_variable.replace("_", " ").title()
                )

                st.divider()

                st.markdown(
                    f"### 📈 Monthly {selected_label} Trend"
                )

                monthly_trend = (
                    year_df.groupby("month")[selected_variable]
                    .mean()
                    .reindex(range(1, 13))
                    .dropna()
                )

                if monthly_trend.empty:

                    st.info(
                        "No monthly values are available for this selection."
                    )

                else:

                    monthly_trend_df = (
                        monthly_trend.rename("value")
                        .rename_axis("month_number")
                        .reset_index()
                    )

                    monthly_trend_df["Month"] = (
                        monthly_trend_df["month_number"].map(
                            lambda month:
                            month_short_names[int(month) - 1]
                        )
                    )

                    monthly_figure = px.line(
                        monthly_trend_df,
                        x="Month",
                        y="value",
                        markers=True,
                        category_orders={
                            "Month": month_short_names
                        },
                        labels={
                            "Month": "Month",
                            "value": selected_label
                        }
                    )

                    monthly_figure.update_layout(
                        height=350,
                        margin=dict(
                            l=10,
                            r=10,
                            t=20,
                            b=10
                        ),
                        xaxis_title="Month",
                        yaxis_title=selected_label
                    )

                    st.plotly_chart(
                        monthly_figure,
                        use_container_width=True
                    )

                st.divider()

                st.markdown("### 📊 Monthly Weather Summary")

                if selected_variable in [
                    "temperature_2m",
                    "relative_humidity_2m"
                ]:

                    monthly_summary = (
                        year_df.groupby("month")[selected_variable]
                        .agg(["mean", "max"])
                        .rename(
                            columns={
                                "mean": "Monthly Average",
                                "max": "Monthly Maximum"
                            }
                        )
                    )

                else:

                    monthly_summary = (
                        year_df.groupby("month")[selected_variable]
                        .agg(["sum", "max"])
                        .rename(
                            columns={
                                "sum": "Monthly Total",
                                "max": "Maximum Hourly Value"
                            }
                        )
                    )

                monthly_summary = monthly_summary.reindex(
                    range(1, 13)
                )

                monthly_summary.index = [
                    month_names[int(month) - 1]
                    for month in monthly_summary.index
                ]

                monthly_summary.index.name = "Month"

                st.dataframe(
                    monthly_summary.round(2),
                    use_container_width=True,
                    height=450
                )

                st.caption(
                    f"Summary statistics for {selected_label} "
                    f"in {selected_year}."
                )

                st.divider()

                st.markdown("### Daily Weather Pattern")

                daily_data = (
                    year_df.set_index("time")[numeric_columns]
                    .resample("D")
                    .mean()
                    .reset_index()
                )

                if selected_variable in daily_data.columns:

                    daily_figure = px.line(
                        daily_data,
                        x="time",
                        y=selected_variable,
                        labels={
                            "time": "Date",
                            selected_variable: selected_label
                        }
                    )

                    daily_figure.update_layout(
                        height=350,
                        margin=dict(
                            l=10,
                            r=10,
                            t=20,
                            b=10
                        ),
                        xaxis=dict(
                            title="Date",
                            tickformat="%b %d",
                            dtick="M1"
                        ),
                        yaxis_title=selected_label
                    )

                    st.plotly_chart(
                        daily_figure,
                        use_container_width=True
                    )

                st.divider()

                st.markdown("### 💡 Weather Insights")

                insight_cols = st.columns(2)

                if "temperature_2m" in numeric_columns:

                    monthly_temperature = (
                        year_df.groupby("month")[
                            "temperature_2m"
                        ]
                        .mean()
                        .dropna()
                    )

                    if not monthly_temperature.empty:

                        warmest_month = int(
                            monthly_temperature.idxmax()
                        )

                        coolest_month = int(
                            monthly_temperature.idxmin()
                        )

                        insight_cols[0].info(
                            f"Warmest month: "
                            f"**{month_names[warmest_month - 1]}** "
                            f"({monthly_temperature.max():.1f} °C)."
                        )

                        insight_cols[1].info(
                            f"Coolest month: "
                            f"**{month_names[coolest_month - 1]}** "
                            f"({monthly_temperature.min():.1f} °C)."
                        )

                if "precipitation" in numeric_columns:

                    monthly_rainfall = (
                        year_df.groupby("month")[
                            "precipitation"
                        ]
                        .sum()
                        .dropna()
                    )

                    if not monthly_rainfall.empty:

                        wettest_month = int(
                            monthly_rainfall.idxmax()
                        )

                        st.info(
                            f"Highest total precipitation: "
                            f"**{month_names[wettest_month - 1]}** "
                            f"({monthly_rainfall.max():.1f} mm)."
                        )

                st.divider()

                st.markdown("### Weather Dataset Preview")

                st.dataframe(
                    year_df.drop(
                        columns=["year", "month"],
                        errors="ignore"
                    ).head(100),
                    use_container_width=True,
                    hide_index=True
                )

                st.download_button(
                    "⬇️ Download Selected Year Weather Data",
                    data=year_df.drop(
                        columns=["year", "month"],
                        errors="ignore"
                    ).to_csv(index=False),
                    file_name=f"hyderabad_weather_{selected_year}.csv",
                    mime="text/csv"
                )

                st.caption(
                    "Historical hourly weather from Open-Meteo Archive "
                    "at a Hyderabad reference coordinate. It is not "
                    "measured at individual accident locations and does "
                    "not establish accident causation. Live traffic data "
                    "is not currently connected."
                )


# ============================================================
# 17. REPORTS AND INSIGHTS
# ============================================================

elif page == "Reports & Insights":

    st.markdown(
        '<div class="section-title">'
        '📑 Reports & Insights'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Download Project Datasets")

    if selected_city == "Hyderabad" and not accidents.empty:

        st.download_button(
            "Download Hyderabad Accident Dataset",
            data=accidents.to_csv(index=False),
            file_name="hyderabad_accidents.csv",
            mime="text/csv"
        )

    if not road_summary.empty:

        st.download_button(
            f"Download {selected_city} Road Type Summary",
            data=road_summary.to_csv(index=False),
            file_name=f"{selected_city.lower()}_road_summary.csv",
            mime="text/csv"
        )

    if selected_city == "Hyderabad" and not weather.empty:

        st.download_button(
            "Download Hyderabad Weather Dataset",
            data=weather.to_csv(index=False),
            file_name="hyderabad_weather.csv",
            mime="text/csv"
        )

    st.divider()

    st.markdown("### Road Segment CSV Export")

    st.warning(
        "The original road CSV files are large. Exporting a complete "
        "road segment file can use significant memory and may take time. "
        "It is not loaded unless you request it below."
    )

    if get_city_csv(selected_city).exists():

        if st.button(
            f"Prepare {selected_city} Road Segment CSV",
            type="secondary"
        ):

            with st.spinner(
                f"Reading {selected_city} road CSV. Please wait..."
            ):

                try:

                    export_df = load_road_csv_for_export(
                        selected_city
                    )

                    if not export_df.empty:

                        st.session_state[
                            f"road_export_{selected_city}"
                        ] = export_df.to_csv(index=False)

                        st.success(
                            "Road segment CSV is ready for download."
                        )

                    else:

                        st.error(
                            "The road CSV could not be loaded."
                        )

                except Exception as error:

                    st.error(
                        f"Could not prepare road CSV: {error}"
                    )

        export_key = f"road_export_{selected_city}"

        if export_key in st.session_state:

            st.download_button(
                f"Download {selected_city} Road Segment Data",
                data=st.session_state[export_key],
                file_name=f"{selected_city.lower()}_roads.csv",
                mime="text/csv"
            )

    else:

        st.warning("Road segment CSV not found.")

    st.divider()

    st.markdown("### Project Data Sources")

    st.markdown("""
    1. **OpenStreetMap:** Road network information for the six selected cities.
    2. **BBBike Extract:** Road data download source.
    3. **Open-Meteo Archive:** Historical Hyderabad weather information.
    4. **Road Accident Dataset:** Annual Hyderabad accident counts.
    """)

    st.markdown("### City Data Availability")

    availability_data = []

    for city in CITIES:

        csv_file = get_city_csv(city)
        summary_file = get_city_summary_file(city)
        geojson_file = get_city_geojson(city)

        availability_data.append({
            "City": city,
            "Road CSV": (
                "Available" if csv_file.exists()
                else "Not available"
            ),
            "Road Summary": (
                "Available" if summary_file.exists()
                else "Not available"
            ),
            "Road GeoJSON": (
                "Available" if geojson_file.exists()
                else "Not available"
            ),
            "Accident Data": (
                "Available"
                if city == "Hyderabad" and not accidents.empty
                else "Not available"
            ),
            "Weather Data": (
                "Available"
                if city == "Hyderabad" and not weather.empty
                else "Not available"
            )
        })

    availability_df = pd.DataFrame(availability_data)

    st.dataframe(
        availability_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Only datasets available in the project folder "
        "are displayed for download."
    )


# ============================================================
# 18. FOOTER
# ============================================================

st.divider()

st.caption(
    "SmartRoad AI | Road Safety Intelligence Dashboard | "
    "MCA Data Analytics & Visualization Project"
)
