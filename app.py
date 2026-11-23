
import json
import hashlib
import warnings
import random
from pathlib import Path

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px

from shapely.geometry import shape

try:
    import ijson
except ImportError:
    ijson = None
from streamlit_folium import st_folium
from folium.plugins import Fullscreen, Geocoder, MeasureControl


# ============================================================
# SMARTROAD AI - COMPLETE STREAMLIT APPLICATION
# Six cities | Lightweight category map | City-specific road screening
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
JICA_TRAFFIC_FILE = DATA_DIR / "raw" / "hyderabad_jica_traffic_counts_2011.csv"
JICA_PCU_FILE = DATA_DIR / "raw" / "hyderabad_jica_traffic_pcu_2011.csv"

# Maximum number of road features sent to the browser.
# This is a display limit only. Original datasets are unchanged.
MAX_MAP_FEATURES = 6500


# ============================================================
# CITY CONFIGURATION
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
# ROAD CATEGORY COLORS
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
    "platform": "#9c27b0",
    "corridor": "#7e57c2",
    "raceway": "#f06292",
    "busway": "#26a69a",
    "road": "#ffffff",
    "unknown": "#ffffff"
}


# Illustrative road-hierarchy screening bands (not measured accident risk).
SCREENING_BAND_COLORS = {
    "High priority": "#ef4444",
    "Medium priority": "#f59e0b",
    "Low priority": "#22c55e"
}

SCREENING_BAND_ORDER = [
    "High priority",
    "Medium priority",
    "Low priority"
]


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
    "track": 20,
    "cycleway": 30,
    "footway": 20,
    "path": 15,
    "pedestrian": 20,
    "steps": 10,
    "construction": 30,
    "unknown": 10
}


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.stApp {
    background: #0b1020;
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #111827;
}
.main-title {
    font-size: 43px;
    font-weight: 850;
    color: #f8fafc;
    letter-spacing: -1px;
}
.sub-title {
    color: #94a3b8;
    font-size: 15px;
}
.section-title {
    color: #f8fafc;
    font-size: 27px;
    font-weight: 750;
}
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #19263b, #111c30);
    border: 1px solid #293951;
    padding: 17px;
    border-radius: 13px;
}
div[data-testid="stAlert"] {
    border-radius: 10px;
}
.stButton button, .stDownloadButton button {
    border-radius: 9px;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# FILE PATH HELPERS
# ============================================================

def city_folder(city):
    return DATA_DIR / "processed" / city.lower()


def city_csv(city):
    return city_folder(city) / f"{city.lower()}_roads.csv"


def city_geojson(city):
    return city_folder(city) / f"{city.lower()}_roads.geojson"


def city_summary(city):
    return city_folder(city) / "road_type_summary.csv"


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data(show_spinner=False)
def load_accidents():
    if not ACCIDENT_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(ACCIDENT_FILE)

    if not {"year", "accidents"}.issubset(df.columns):
        return pd.DataFrame()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["accidents"] = pd.to_numeric(
        df["accidents"], errors="coerce"
    )

    return (
        df.dropna(subset=["year", "accidents"])
        .sort_values("year")
        .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def load_weather():
    if not WEATHER_FILE.exists():
        return pd.DataFrame()

    df = pd.read_csv(WEATHER_FILE)

    if "time" in df.columns:
        df["time"] = pd.to_datetime(
            df["time"], errors="coerce"
        )

    return df


@st.cache_data(show_spinner=False)
def load_jica_traffic():
    """Load JICA 2011 directional/movement traffic survey records."""
    if not JICA_TRAFFIC_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(JICA_TRAFFIC_FILE)
        required = {
            "city", "location_name", "survey_date", "survey_duration_hours",
            "count_type", "direction_or_movement", "vehicle_count"
        }
        if not required.issubset(df.columns):
            return pd.DataFrame()

        df["survey_date"] = pd.to_datetime(df["survey_date"], errors="coerce")
        df["survey_duration_hours"] = pd.to_numeric(
            df["survey_duration_hours"], errors="coerce"
        )
        df["vehicle_count"] = pd.to_numeric(df["vehicle_count"], errors="coerce")
        df = df.dropna(subset=["survey_date", "vehicle_count"])
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_jica_pcu():
    """Load location-level PCU totals; legacy 07:00 placeholders are dates only."""
    if not JICA_PCU_FILE.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(JICA_PCU_FILE)
        # The supplied PCU file currently uses the generic traffic-template
        # columns. Its 07:00 timestamp is a placeholder, not an observation time.
        if {"vehicle_count", "road_name", "timestamp"}.issubset(df.columns):
            df = df.rename(columns={
                "vehicle_count": "vehicle_count_pcu",
                "road_name": "location_name"
            })
            stamp = pd.to_datetime(df["timestamp"], errors="coerce")
            df["survey_date"] = stamp.dt.date
            df["count_unit"] = "PCU total"
            df["survey_duration_hours"] = pd.to_numeric(
                df["location_name"].astype(str).str.extract(r"\((\d+(?:\.\d+)?)\s*-?hour", expand=False),
                errors="coerce"
            )
            # Accept the common '(14-hour total; PCU)' notation.
            missing_duration = df["survey_duration_hours"].isna()
            df.loc[missing_duration, "survey_duration_hours"] = pd.to_numeric(
                df.loc[missing_duration, "location_name"].astype(str).str.extract(
                    r"\((\d+(?:\.\d+)?)\s*hour", expand=False
                )[0], errors="coerce"
            )
            df["location_name"] = df["location_name"].astype(str).str.replace(
                r"\s*\([^)]*\)\s*$", "", regex=True
            )
            df["city"] = df.get("city", "Hyderabad")
        elif {"vehicle_count_pcu", "location_name", "survey_duration_hours"}.issubset(df.columns):
            df["survey_duration_hours"] = pd.to_numeric(
                df["survey_duration_hours"], errors="coerce"
            )
        else:
            return pd.DataFrame()

        df["vehicle_count_pcu"] = pd.to_numeric(
            df["vehicle_count_pcu"], errors="coerce"
        )
        df["survey_duration_hours"] = pd.to_numeric(
            df["survey_duration_hours"], errors="coerce"
        )
        df = df.dropna(subset=["vehicle_count_pcu", "survey_duration_hours"])
        df["derived_pcu_per_hour"] = (
            df["vehicle_count_pcu"] / df["survey_duration_hours"]
        )
        return df.reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_road_summary(city):
    path = city_summary(city)

    if not path.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(path)

        required = {
            "road_type",
            "road_segments",
            "total_length_km"
        }

        if not required.issubset(df.columns):
            return pd.DataFrame()

        df["road_segments"] = pd.to_numeric(
            df["road_segments"], errors="coerce"
        ).fillna(0)

        df["total_length_km"] = pd.to_numeric(
            df["total_length_km"], errors="coerce"
        ).fillna(0)

        return df

    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False, max_entries=6)
def load_road_geometry(city, per_category_limit=250):
    """
    Stream the city's processed GeoJSON and retain a deterministic,
    category-balanced sample for map display.

    The full multi-GB GeoJSON is never loaded into memory. Only the
    sampled LineString/MultiLineString geometries are retained.
    """
    empty = gpd.GeoDataFrame(
        columns=["road_type", "length_km", "geometry"],
        geometry="geometry",
        crs="EPSG:4326"
    )

    if ijson is None:
        raise RuntimeError(
            "The 'ijson' package is required for streaming GeoJSON. "
            "Install it with: pip install ijson"
        )

    geojson_path = city_geojson(city)
    if not geojson_path.exists():
        raise FileNotFoundError(
            f"Processed GeoJSON not found for {city}: {geojson_path}"
        )

    rng = random.Random(42)
    reservoirs = {}
    category_counts = {}

    try:
        with geojson_path.open("rb") as source:
            for feature in ijson.items(source, "features.item", use_float=True):
                if not isinstance(feature, dict):
                    continue

                properties = feature.get("properties") or {}
                geometry_data = feature.get("geometry")

                if not geometry_data:
                    continue

                road_type = (
                    properties.get("road_type")
                    or properties.get("highway")
                    or "unknown"
                )
                if isinstance(road_type, (list, dict)):
                    road_type = str(road_type)
                road_type = str(road_type).strip().lower() or "unknown"

                try:
                    geometry = shape(geometry_data)
                except Exception:
                    continue

                if geometry.is_empty or geometry.geom_type not in (
                    "LineString", "MultiLineString"
                ):
                    continue

                try:
                    length_value = properties.get("length_km")
                    if length_value is not None:
                        length_km = float(length_value)
                    else:
                        length_km = float(properties.get("length_m") or 0) / 1000
                except (TypeError, ValueError):
                    length_km = 0.0

                row = {
                    "road_type": road_type,
                    "length_km": length_km,
                    "geometry": geometry
                }

                category_counts[road_type] = (
                    category_counts.get(road_type, 0) + 1
                )
                seen = category_counts[road_type]
                bucket = reservoirs.setdefault(road_type, [])

                if len(bucket) < per_category_limit:
                    bucket.append(row)
                else:
                    replacement_index = rng.randrange(seen)
                    if replacement_index < per_category_limit:
                        bucket[replacement_index] = row

        sampled_rows = [
            row
            for category_rows in reservoirs.values()
            for row in category_rows
        ]

        if not sampled_rows:
            return empty

        if len(sampled_rows) > MAX_MAP_FEATURES:
            sampled_rows = rng.sample(sampled_rows, MAX_MAP_FEATURES)

        result = gpd.GeoDataFrame(
            sampled_rows,
            geometry="geometry",
            crs="EPSG:4326"
        )

        return result.reset_index(drop=True)

    except Exception as exc:
        raise RuntimeError(
            f"Could not stream the processed GeoJSON for {city}: {exc}"
        ) from exc


# ============================================================
# ROAD SUMMARY HELPERS
# ============================================================

def screening_band(weight):
    """Convert an illustrative road-hierarchy weight into a screening band."""
    try:
        value = float(weight)
    except (TypeError, ValueError):
        value = 10.0

    if value >= 75:
        return "High priority"
    if value >= 40:
        return "Medium priority"
    return "Low priority"


def city_screening_comparison(summaries):
    """Build a city comparison from each city's own road summary."""
    rows = []

    for city_name, city_summary_data in summaries.items():
        if city_summary_data.empty:
            continue

        data = city_summary_data.copy()
        data["screening_weight"] = (
            data["road_type"].map(ROAD_PRIORITY_WEIGHTS).fillna(10)
        )
        segment_total = float(data["road_segments"].sum())
        if segment_total <= 0:
            continue

        weighted_index = (
            data["screening_weight"] * data["road_segments"]
        ).sum() / segment_total

        data["screening_band"] = data["screening_weight"].apply(screening_band)
        high_priority_segments = data.loc[
            data["screening_band"] == "High priority", "road_segments"
        ].sum()

        rows.append({
            "City": city_name,
            "Road Segments": int(segment_total),
            "Mapped Length (km)": float(data["total_length_km"].sum()),
            "Road Categories": int(data["road_type"].nunique()),
            "Illustrative Screening Index": round(float(weighted_index), 2),
            "High-Priority Class Segments": int(high_priority_segments)
        })

    return pd.DataFrame(rows)


def summary_metrics(summary):
    if summary.empty:
        return 0, 0.0, 0

    return (
        int(summary["road_segments"].sum()),
        float(summary["total_length_km"].sum()),
        int(summary["road_type"].nunique())
    )


def make_map_sample(roads, limit=MAX_MAP_FEATURES):
    """
    Keep a balanced sample across road categories.
    This affects map display only, not the source dataset.
    """

    if len(roads) <= limit:
        return roads.copy()

    categories = roads["road_type"].value_counts()
    sampled_parts = []

    total = max(int(categories.sum()), 1)

    for category, count in categories.items():
        share = max(1, int(limit * count / total))

        group = roads[roads["road_type"] == category]

        sampled_parts.append(
            group.sample(
                n=min(len(group), share),
                random_state=42
            )
        )

    sample = pd.concat(sampled_parts, ignore_index=True)

    if len(sample) > limit:
        sample = sample.sample(
            n=limit,
            random_state=42
        )

    return gpd.GeoDataFrame(
        sample,
        geometry="geometry",
        crs="EPSG:4326"
    )


# ============================================================
# LOAD LIGHTWEIGHT DATA
# ============================================================

accidents = load_accidents()
weather = load_weather()
jica_traffic = load_jica_traffic()
jica_pcu = load_jica_pcu()

# Load only summary CSVs. Do not load city geometry here.
ALL_SUMMARIES = {
    city: load_road_summary(city)
    for city in CITIES
}


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 🚦 SMARTROAD AI")
    st.caption("SMART CITY ROAD INTELLIGENCE")

    st.divider()

    page = st.radio(
        "Navigation",
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

    selected_summary = ALL_SUMMARIES[selected_city]

    if selected_summary.empty:
        st.warning("Road summary not found.")
    else:
        st.success("Road summary available")

    st.divider()

    st.success("Dashboard online")

    st.caption(
        "Road network data: OpenStreetMap / BBBike"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 SmartRoad AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Smart City Road Safety, Infrastructure & Traffic Intelligence'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 1. OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="section-title">City Intelligence Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Currently exploring **{selected_city}, "
        f"{city_info['state']}**."
    )

    summary = selected_summary
    road_count, road_length, category_count = summary_metrics(
        summary
    )

    cities_with_roads = sum(
        not value.empty for value in ALL_SUMMARIES.values()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Road Segments", f"{road_count:,}")
    c2.metric("Mapped Length", f"{road_length:,.1f} km")
    c3.metric("Road Categories", category_count)
    c4.metric("Cities With Road Data", cities_with_roads)

    st.divider()

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("### Road Category Distribution")

        if not summary.empty:
            fig = px.bar(
                summary.sort_values(
                    "road_segments",
                    ascending=False
                ),
                x="road_type",
                y="road_segments",
                color="road_type",
                color_discrete_map=ROAD_COLORS,
                title=f"{selected_city} - Road Segments"
            )

            fig.update_layout(
                height=380,
                showlegend=False,
                xaxis_title="Road Category",
                yaxis_title="Segments"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No road summary available.")

    with right:
        st.markdown("### Road Length by Category")

        if not summary.empty:
            fig = px.pie(
                summary,
                names="road_type",
                values="total_length_km",
                hole=0.48,
                color="road_type",
                color_discrete_map=ROAD_COLORS
            )

            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

    if not summary.empty:
        st.markdown("### Road Network Summary")

        display = summary.rename(
            columns={
                "road_type": "Road Category",
                "road_segments": "Road Segments",
                "total_length_km": "Mapped Length (km)"
            }
        )

        display["Mapped Length (km)"] = (
            display["Mapped Length (km)"].round(2)
        )

        st.dataframe(
            display,
            use_container_width=True,
            hide_index=True
        )

    st.caption(
        "Mapped lengths include OpenStreetMap highway features such "
        "as service roads, footways and paths. They are not necessarily "
        "official motor-vehicle road mileage."
    )

    st.divider()

    st.markdown(f"### {selected_city} Accident Data Availability")

    if selected_city == "Hyderabad" and not accidents.empty:
        overview_accidents = accidents.copy()
        overview_accidents["Year"] = pd.to_numeric(
            overview_accidents["year"], errors="coerce"
        )
        overview_accidents["Accidents"] = pd.to_numeric(
            overview_accidents["accidents"], errors="coerce"
        )
        overview_accidents = overview_accidents.dropna(
            subset=["Year", "Accidents"]
        ).sort_values("Year")
        overview_accidents["Year"] = overview_accidents["Year"].astype(int).astype(str)

        fig = px.bar(
            overview_accidents,
            x="Year",
            y="Accidents",
            text="Accidents",
            title="Hyderabad - Annual Reported Accident Counts",
            labels={"Year": "Year", "Accidents": "Reported Accidents"}
        )
        fig.update_traces(
            texttemplate="%{text:,.0f}",
            textposition="outside",
            cliponaxis=False
        )
        fig.update_layout(
            height=420,
            xaxis={"type": "category", "title": "Year", "categoryorder": "array",
                   "categoryarray": overview_accidents["Year"].tolist()},
            yaxis={"title": "Number of Accidents", "rangemode": "tozero"},
            showlegend=False,
            margin={"t": 70, "b": 45}
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "These annual city-level records are available for Hyderabad only; "
            "they do not provide individual crash locations."
        )
    elif selected_city == "Hyderabad":
        st.info("Hyderabad accident dataset is not available.")
    else:
        st.info(
            f"Accident records for {selected_city} are not currently included. "
            "This dashboard will not substitute Hyderabad figures for this city."
        )


# ============================================================
# 2. INTERACTIVE MAP
# ============================================================

elif page == "Interactive Map":

    st.markdown(
        '<div class="section-title">🗺️ Interactive Road Intelligence Map</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Explore the mapped road network of **{selected_city}**."
    )

    with st.spinner(
        f"Building a lightweight road map for {selected_city}..."
    ):
        try:
            roads = load_road_geometry(selected_city)
            map_load_error = None
        except Exception as exc:
            roads = gpd.GeoDataFrame(
                columns=["road_type", "length_km", "geometry"],
                geometry="geometry", crs="EPSG:4326"
            )
            map_load_error = str(exc)

    if roads is None or roads.empty:

        st.error(f"No usable lightweight road geometry found for {selected_city}.")
        if map_load_error:
            st.code(map_load_error)
        st.info(
            "The map reads the selected city's processed GeoJSON using a "
            "streaming parser. Check that the file exists and contains a "
            "FeatureCollection with LineString or MultiLineString road features."
        )

    else:

        # ----------------------------------------------------
        # MAP CONTROLS
        # ----------------------------------------------------

        c1, c2, c3 = st.columns([1.1, 1.1, 1])

        with c1:
            map_style = st.selectbox(
                "Map Style",
                ["Street Map", "Dark Map", "Satellite"],
                key="road_map_style"
            )

        with c2:
            display_mode = st.selectbox(
                "Road Display",
                ["All Categories", "Major Roads", "Choose Categories"],
                key="road_map_display_mode"
            )

        with c3:
            show_marker = st.checkbox(
                "Show City Centre",
                value=True,
                key="road_map_city_marker"
            )

        available_types = sorted(
            roads["road_type"].dropna().unique().tolist()
        )

        if display_mode == "Major Roads":
            major_types = [
                "motorway", "motorway_link", "trunk", "trunk_link",
                "primary", "primary_link", "secondary", "secondary_link",
                "tertiary", "tertiary_link"
            ]
            selected_types = [
                item for item in major_types if item in available_types
            ]
        elif display_mode == "Choose Categories":
            selected_types = st.multiselect(
                "Select road categories",
                options=available_types,
                default=[
                    item for item in [
                        "motorway", "trunk", "primary", "secondary",
                        "tertiary", "residential"
                    ] if item in available_types
                ],
                key="road_map_category_filter"
            )
        else:
            selected_types = available_types

        color_mode = st.radio(
            "Colour roads by",
            ["Screening Priority", "Road Category"],
            horizontal=True,
            key="road_map_color_mode"
        )

        selected_bands = st.multiselect(
            "Filter screening priority",
            options=SCREENING_BAND_ORDER,
            default=SCREENING_BAND_ORDER,
            key="road_map_priority_filter"
        )

        roads = roads.copy()
        roads["screening_weight"] = (
            roads["road_type"].map(ROAD_PRIORITY_WEIGHTS).fillna(10)
        )
        roads["screening_band"] = roads["screening_weight"].apply(screening_band)

        filtered = roads[
            roads["road_type"].isin(selected_types)
            & roads["screening_band"].isin(selected_bands)
        ].copy()

        # ----------------------------------------------------
        # MAP DISPLAY SAMPLE
        # ----------------------------------------------------

        # Geometry dataframe is already a small representative sample.
        display_roads = filtered.copy()
        displayed_count = len(display_roads)
        summary_for_map = selected_summary.copy()
        if summary_for_map.empty:
            total_available = 0
            total_filtered = 0
        else:
            summary_for_map["screening_weight"] = (
                summary_for_map["road_type"].map(ROAD_PRIORITY_WEIGHTS).fillna(10)
            )
            summary_for_map["screening_band"] = (
                summary_for_map["screening_weight"].apply(screening_band)
            )
            total_available = int(summary_for_map["road_segments"].sum())
            total_filtered = int(summary_for_map.loc[
                summary_for_map["road_type"].isin(selected_types)
                & summary_for_map["screening_band"].isin(selected_bands),
                "road_segments"
            ].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Available Road Segments", f"{total_available:,}")
        c2.metric("Selected Segments", f"{total_filtered:,}")
        c3.metric("Map Lines Displayed", f"{displayed_count:,}")

        st.info(
            f"For fast rendering, the map displays {displayed_count:,} "
            f"representative road lines from {total_filtered:,} selected "
            "source segments. Full counts and lengths come from the city's "
            "road summary. Priority colours are illustrative road-hierarchy "
            "screening bands, not measured accident risk."
        )

        # ----------------------------------------------------
        # CREATE MAP
        # ----------------------------------------------------

        m = folium.Map(
            location=[latitude, longitude],
            zoom_start=12,
            tiles=None,
            control_scale=True,
            prefer_canvas=True
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

        # ----------------------------------------------------
        # SEPARATE CATEGORY LAYERS
        # ----------------------------------------------------

        for road_type in selected_types:

            category_roads = display_roads[
                display_roads["road_type"] == road_type
            ]

            if category_roads.empty:
                continue

            if color_mode == "Screening Priority":
                band_values = category_roads["screening_band"].dropna().unique()
                color = SCREENING_BAND_COLORS.get(
                    band_values[0] if len(band_values) else "Low priority",
                    "#94a3b8"
                )
            else:
                color = ROAD_COLORS.get(road_type, "#ffffff")

            layer = folium.FeatureGroup(
                name=road_type.replace(
                    "_", " "
                ).title(),
                show=True
            )

            geojson = json.loads(
                category_roads[
                    ["road_type", "length_km", "screening_weight",
                     "screening_band", "geometry"]
                ].to_json()
            )

            folium.GeoJson(
                geojson,
                name=road_type,
                style_function=lambda feature, c=color: {
                    "color": (
                        SCREENING_BAND_COLORS.get(
                            feature["properties"].get("screening_band"),
                            "#94a3b8"
                        ) if color_mode == "Screening Priority" else c
                    ),
                    "weight": (
                        4 if feature["properties"].get("road_type") in [
                            "motorway", "trunk", "primary"
                        ] else 2
                    ),
                    "opacity": 0.9
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=[
                        "road_type", "length_km",
                        "screening_band", "screening_weight"
                    ],
                    aliases=[
                        "Road Category:",
                        "Length (km):",
                        "Screening Priority:",
                        "Illustrative Weight:"
                    ],
                    localize=True
                )
            ).add_to(layer)

            layer.add_to(m)

        if show_marker:
            folium.Marker(
                [latitude, longitude],
                tooltip=f"{selected_city} City Centre",
                popup=f"{selected_city}, {city_info['state']}",
                icon=folium.Icon(
                    color="red",
                    icon="info-sign"
                )
            ).add_to(m)

        # ----------------------------------------------------
        # MAP LEGEND
        # ----------------------------------------------------

        if color_mode == "Screening Priority":
            legend_html = """
            <div style="
                position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: #ffffff; color: #111827; padding: 12px 15px;
                border-radius: 9px; border: 1px solid #cbd5e1;
                font-size: 12px; box-shadow: 0 2px 8px rgba(0,0,0,.25);">
                <b>ILLUSTRATIVE SCREENING PRIORITY</b><br>
                <div style="margin-top:6px;"><span style="color:#ef4444;">━━</span> High priority (75–100)</div>
                <div style="margin-top:4px;"><span style="color:#f59e0b;">━━</span> Medium priority (40–74)</div>
                <div style="margin-top:4px;"><span style="color:#22c55e;">━━</span> Low priority (below 40)</div>
            </div>
            """
        else:
            legend_html = """
            <div style="
                position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: #ffffff; color: #111827; padding: 12px 15px;
                border-radius: 9px; border: 1px solid #cbd5e1;
                max-height: 260px; overflow-y: auto; font-size: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,.25);">
                <b>ROAD CATEGORY LEGEND</b><br>
            """
            for road_type in selected_types:
                color = ROAD_COLORS.get(road_type, "#ffffff")
                label = road_type.replace("_", " ").title()
                legend_html += f"""
                <div style="margin-top:5px;">
                    <span style="display:inline-block;width:19px;height:4px;
                        background:{color};margin-right:8px;vertical-align:middle;"></span>
                    {label}
                </div>
                """
            legend_html += "</div>"

        m.get_root().html.add_child(
            folium.Element(legend_html)
        )

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
            collapsed=False
        ).add_to(m)

        st_folium(
            m,
            width=None,
            height=690,
            returned_objects=[]
        )

        st.caption(
            "Source: OpenStreetMap / BBBike processed road network. "
            "Lines are a category-balanced representative sample, not every "
            "source segment. Priority colours are illustrative road-hierarchy "
            "screening bands, not measured accident risk or accident locations."
        )


# ============================================================
# 3. ACCIDENT ANALYSIS
# ============================================================

elif page == "Accident Analysis":

    st.markdown(
        '<div class="section-title">📊 Road Accident Analysis</div>',
        unsafe_allow_html=True
    )

    if selected_city != "Hyderabad":

        st.warning(
            "Verified accident records in the current project "
            "are available for Hyderabad only."
        )

    elif accidents.empty:

        st.error("Accident dataset not found.")

    else:

        years = sorted(
            accidents["year"].astype(int).unique().tolist()
        )

        if len(years) >= 2:
            selected_years = st.select_slider(
                "Analysis Period",
                options=years,
                value=(years[0], years[-1])
            )

            data = accidents[
                accidents["year"].between(
                    selected_years[0],
                    selected_years[1]
                )
            ].copy()
        else:
            data = accidents.copy()

        if data.empty:
            st.warning("No records for this period.")
        else:

            total = data["accidents"].sum()
            average = data["accidents"].mean()

            high = data.loc[
                data["accidents"].idxmax()
            ]

            low = data.loc[
                data["accidents"].idxmin()
            ]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric("Total Accidents", f"{total:,.0f}")
            c2.metric("Yearly Average", f"{average:,.0f}")
            c3.metric(
                "Highest Year",
                str(int(high["year"])),
                help=f"{high['accidents']:,.0f} accidents"
            )
            c4.metric(
                "Lowest Year",
                str(int(low["year"])),
                help=f"{low['accidents']:,.0f} accidents"
            )

            st.divider()

            left, right = st.columns(2)

            with left:
                fig = px.line(
                    data,
                    x="year",
                    y="accidents",
                    markers=True,
                    title="Annual Accident Trend"
                )
                fig.update_layout(height=370)
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with right:
                fig = px.bar(
                    data,
                    x="year",
                    y="accidents",
                    color="accidents",
                    title="Year-wise Accident Comparison"
                )
                fig.update_layout(height=370)
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            st.markdown("### Year-on-Year Change")

            comparison = data.copy()
            comparison["change"] = (
                comparison["accidents"].diff()
            )
            comparison["change_percent"] = (
                comparison["accidents"].pct_change() * 100
            )

            st.dataframe(
                comparison.round(2),
                use_container_width=True,
                hide_index=True
            )

            if len(data) >= 2:
                first = data.iloc[0]
                last = data.iloc[-1]

                if first["accidents"] != 0:
                    change = (
                        (last["accidents"] - first["accidents"])
                        / first["accidents"]
                    ) * 100

                    if change > 0:
                        st.warning(
                            f"Recorded accidents increased by "
                            f"{change:.2f}% between "
                            f"{int(first['year'])} and "
                            f"{int(last['year'])}."
                        )
                    elif change < 0:
                        st.success(
                            f"Recorded accidents decreased by "
                            f"{abs(change):.2f}% between "
                            f"{int(first['year'])} and "
                            f"{int(last['year'])}."
                        )
                    else:
                        st.info("No change in the selected endpoints.")

            st.markdown("### Accident Dataset")

            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Accident Data",
                data=data.to_csv(index=False),
                file_name="hyderabad_accident_analysis.csv",
                mime="text/csv"
            )

            st.caption(
                "These are annual city-level counts. They do not "
                "identify individual crash locations or establish causes."
            )


# ============================================================
# 4. ROAD RISK INTELLIGENCE
# ============================================================

elif page == "Road Risk Intelligence":

    st.markdown(
        f'<div class="section-title">⚠️ {selected_city} Infrastructure Screening</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Six-City Road-Hierarchy Comparison")
    comparison = city_screening_comparison(ALL_SUMMARIES)

    if comparison.empty:
        st.info("No city road summaries are available for comparison.")
    else:
        st.caption(
            "Comparison uses each city's own road-category summary. "
            "The index is a segment-count-weighted average of illustrative "
            "road-hierarchy weights; it is not an accident-risk score."
        )
        comparison_chart = px.bar(
            comparison.sort_values(
                "Illustrative Screening Index", ascending=False
            ),
            x="City",
            y="Illustrative Screening Index",
            color="Illustrative Screening Index",
            color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
            hover_data=[
                "Road Segments", "Mapped Length (km)",
                "Road Categories", "High-Priority Class Segments"
            ],
            title="Illustrative Road-Hierarchy Screening Index by City"
        )
        comparison_chart.update_layout(height=380, coloraxis_showscale=False)
        st.plotly_chart(comparison_chart, use_container_width=True)
        st.dataframe(
            comparison.round(2),
            use_container_width=True,
            hide_index=True
        )
        st.download_button(
            "⬇️ Download Six-City Screening Comparison",
            data=comparison.to_csv(index=False).encode("utf-8"),
            file_name="smartroad_six_city_screening_comparison.csv",
            mime="text/csv",
            key="six_city_screening_comparison_download"
        )

    st.divider()
    summary = selected_summary

    if summary.empty:

        st.warning(
            f"Road summary is not available for {selected_city}."
        )

    else:

        analysis = summary.copy()

        analysis["screening_weight"] = (
            analysis["road_type"]
            .map(ROAD_PRIORITY_WEIGHTS)
            .fillna(10)
        )
        analysis["screening_band"] = analysis["screening_weight"].apply(
            screening_band
        )

        total_segments = analysis["road_segments"].sum()
        total_length = analysis["total_length_km"].sum()

        analysis["segment_share_percent"] = (
            analysis["road_segments"] /
            max(total_segments, 1) * 100
        )

        analysis["length_share_percent"] = (
            analysis["total_length_km"] /
            max(total_length, 1) * 100
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Mapped Segments",
            f"{total_segments:,.0f}"
        )

        c2.metric(
            "Mapped Length",
            f"{total_length:,.1f} km"
        )

        c3.metric(
            "Road Categories",
            analysis["road_type"].nunique()
        )

        c4.metric(
            "Major Road Classes",
            analysis[
                analysis["road_type"].isin(
                    [
                        "motorway",
                        "trunk",
                        "primary",
                        "secondary"
                    ]
                )
            ]["road_type"].nunique()
        )

        st.divider()

        st.info(
            f"This screening is calculated from {selected_city}'s own road "
            "category summary. High/medium/low bands are based on illustrative "
            "road-hierarchy weights (high: 75–100, medium: 40–74, low: below 40). "
            "They are not accident-risk probabilities, validated hotspots or "
            "machine-learning predictions. Hyderabad accident counts are not "
            "used in this city screening."
        )

        band_totals = (
            analysis.groupby("screening_band")["road_segments"].sum()
            .reindex(SCREENING_BAND_ORDER, fill_value=0)
        )
        band_cols = st.columns(3)
        band_cols[0].metric("High-Priority Class Segments", f"{band_totals['High priority']:,.0f}")
        band_cols[1].metric("Medium-Priority Class Segments", f"{band_totals['Medium priority']:,.0f}")
        band_cols[2].metric("Low-Priority Class Segments", f"{band_totals['Low priority']:,.0f}")

        st.caption(
            "These counts classify road categories using illustrative hierarchy "
            "weights. They do not identify accident hotspots or unsafe individual roads."
        )

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            selected_types = st.multiselect(
                "Filter Road Categories",
                options=sorted(analysis["road_type"].unique()),
                default=sorted(analysis["road_type"].unique()),
                key=f"risk_types_{selected_city}"
            )
        with filter_col2:
            selected_bands_risk = st.multiselect(
                "Filter Screening Priority",
                options=SCREENING_BAND_ORDER,
                default=SCREENING_BAND_ORDER,
                key=f"risk_bands_{selected_city}"
            )

        filtered = analysis[
            analysis["road_type"].isin(selected_types)
            & analysis["screening_band"].isin(selected_bands_risk)
        ].copy()

        if filtered.empty:
            st.warning("Select at least one road category.")
        else:

            left, right = st.columns(2)

            with left:
                fig = px.bar(
                    filtered.sort_values(
                        "road_segments",
                        ascending=False
                    ),
                    x="road_type",
                    y="road_segments",
                    color="road_type",
                    color_discrete_map=ROAD_COLORS,
                    title="Road Segment Distribution"
                )
                fig.update_layout(
                    height=380,
                    showlegend=False
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            with right:
                fig = px.bar(
                    filtered.sort_values(
                        "total_length_km",
                        ascending=False
                    ),
                    x="road_type",
                    y="total_length_km",
                    color="road_type",
                    color_discrete_map=ROAD_COLORS,
                    title="Mapped Road Length"
                )
                fig.update_layout(
                    height=380,
                    showlegend=False
                )
                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            fig = px.bar(
                filtered.sort_values(
                    "screening_weight",
                    ascending=False
                ),
                x="road_type",
                y="screening_weight",
                color="road_type",
                color_discrete_map=ROAD_COLORS,
                title="Illustrative Infrastructure Screening Weights"
            )

            fig.update_layout(
                height=350,
                showlegend=False,
                yaxis_range=[0, 110]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            display = filtered.rename(
                columns={
                    "road_type": "Road Category",
                    "road_segments": "Road Segments",
                    "total_length_km": "Mapped Length (km)",
                    "screening_weight": "Illustrative Weight",
                    "screening_band": "Screening Priority",
                    "segment_share_percent": "Segment Share (%)",
                    "length_share_percent": "Length Share (%)"
                }
            )

            st.markdown("### Infrastructure Screening Table")

            st.dataframe(
                display.round(2),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "⬇️ Download Screening Report",
                data=filtered.to_csv(index=False),
                file_name=(
                    f"{selected_city.lower()}_"
                    "infrastructure_screening.csv"
                ),
                mime="text/csv"
            )

        st.divider()

        st.markdown("### Interpretation")

        st.markdown("""
        - **Road segments:** Count of mapped road features.
        - **Mapped length:** Total length recorded in the road summary.
        - **Illustrative weight:** Road-hierarchy value assigned to each class.
        - **Screening priority:** High, medium or low band derived from that illustrative weight.
        - **Segment share:** Percentage of road features in each category.
        - **Length share:** Percentage of mapped length in each category.

        Actual accident-risk modelling requires verified accident locations,
        crash severity, traffic exposure and road-condition information.
        """)


# ============================================================
# 5. TRAFFIC & WEATHER
# ============================================================

elif page == "Traffic & Weather":

    st.markdown(
        '<div class="section-title">🚦 Traffic & Weather Intelligence</div>',
        unsafe_allow_html=True
    )

    st.write(
        f"Explore traffic observations and historical weather for **{selected_city}**."
    )

    st.info(
        "Traffic analysis uses a CSV dataset uploaded by you. This app does not "
        "currently connect to a live traffic API, so uploaded observations must "
        "not be interpreted as live traffic."
    )

    # --------------------------------------------------------
    # TRAFFIC DATA: TEMPLATE, UPLOAD AND VALIDATION
    # --------------------------------------------------------

    st.markdown("### 🚗 Traffic Data Analysis")
    st.write(
        "Upload traffic observations for one or more of the six supported cities. "
        "The app will filter the records for the city selected in the sidebar."
    )

    traffic_template = pd.DataFrame({
        "city": ["Hyderabad", "Bengaluru", "Mumbai"],
        "timestamp": [
            "2026-01-01 08:00:00",
            "2026-01-01 08:00:00",
            "2026-01-01 08:00:00"
        ],
        "vehicle_count": [120, 95, 150],
        "average_speed_kmph": [28.5, 32.0, 21.5],
        "road_name": ["Example Road 1", "Example Road 2", "Example Road 3"]
    })

    st.download_button(
        "⬇️ Download Traffic CSV Template",
        data=traffic_template.to_csv(index=False),
        file_name="smartroad_traffic_template.csv",
        mime="text/csv",
        key="traffic_template_download"
    )

    st.caption(
        "Required columns: city, timestamp, vehicle_count, average_speed_kmph. "
        "Optional column: road_name. Replace the example rows with your actual "
        "traffic observations before uploading."
    )

    traffic_file = st.file_uploader(
        "Upload Traffic Dataset (CSV)",
        type=["csv"],
        key="traffic_csv_uploader",
        help="Upload observed traffic data. Example values in the template are placeholders, not real measurements."
    )

    if traffic_file is None:
        st.warning(
            "No traffic dataset uploaded. Download the template, add verified "
            "traffic observations, and upload the completed CSV to view traffic analysis."
        )
        traffic_data = pd.DataFrame()
    else:
        try:
            traffic_data = pd.read_csv(traffic_file)
            traffic_data.columns = [str(col).strip().lower() for col in traffic_data.columns]
            required_traffic_columns = {
                "city", "timestamp", "vehicle_count", "average_speed_kmph"
            }
            missing_columns = required_traffic_columns - set(traffic_data.columns)

            if missing_columns:
                st.error(
                    "Traffic CSV is missing required columns: "
                    + ", ".join(sorted(missing_columns))
                )
                traffic_data = pd.DataFrame()
            else:
                traffic_data["city"] = traffic_data["city"].astype(str).str.strip()
                traffic_data["timestamp"] = pd.to_datetime(
                    traffic_data["timestamp"], errors="coerce"
                )
                traffic_data["vehicle_count"] = pd.to_numeric(
                    traffic_data["vehicle_count"], errors="coerce"
                )
                traffic_data["average_speed_kmph"] = pd.to_numeric(
                    traffic_data["average_speed_kmph"], errors="coerce"
                )

                valid_cities = list(CITIES.keys())
                valid_rows = (
                    traffic_data["city"].isin(valid_cities)
                    & traffic_data["timestamp"].notna()
                    & traffic_data["vehicle_count"].notna()
                    & traffic_data["average_speed_kmph"].notna()
                    & (traffic_data["vehicle_count"] >= 0)
                    & (traffic_data["average_speed_kmph"] >= 0)
                )
                rejected_rows = int((~valid_rows).sum())
                traffic_data = traffic_data.loc[valid_rows].copy()

                if rejected_rows:
                    st.warning(
                        f"Skipped {rejected_rows:,} invalid row(s). Check city names, "
                        "timestamps and non-negative numeric traffic values."
                    )

                if traffic_data.empty:
                    st.error("No valid traffic observations were found in the uploaded CSV.")
                else:
                    st.success(
                        f"Loaded {len(traffic_data):,} valid traffic observation(s)."
                    )
        except Exception as exc:
            st.error(f"Could not read the uploaded traffic CSV: {exc}")
            traffic_data = pd.DataFrame()

    if not traffic_data.empty:
        city_traffic = traffic_data[
            traffic_data["city"].str.casefold() == selected_city.casefold()
        ].copy()

        st.markdown(f"#### {selected_city} Traffic Summary")

        if city_traffic.empty:
            st.info(
                f"The uploaded file does not contain valid traffic observations "
                f"for {selected_city}. It may contain records for another supported city."
            )
        else:
            city_traffic = city_traffic.sort_values("timestamp")
            c1, c2, c3 = st.columns(3)
            c1.metric("Traffic Records", f"{len(city_traffic):,}")
            c2.metric(
                "Vehicles Recorded",
                f"{city_traffic['vehicle_count'].sum():,.0f}"
            )
            c3.metric(
                "Average Recorded Speed",
                f"{city_traffic['average_speed_kmph'].mean():.1f} km/h"
            )

            daily_traffic = (
                city_traffic.set_index("timestamp")
                .resample("D")
                .agg({
                    "vehicle_count": "sum",
                    "average_speed_kmph": "mean"
                })
                .dropna(how="all")
                .reset_index()
            )

            left, right = st.columns(2)
            with left:
                fig = px.line(
                    daily_traffic,
                    x="timestamp",
                    y="vehicle_count",
                    markers=True,
                    title="Daily Recorded Vehicle Count",
                    labels={
                        "timestamp": "Date",
                        "vehicle_count": "Vehicles Recorded"
                    }
                )
                fig.update_layout(height=370)
                st.plotly_chart(fig, use_container_width=True)

            with right:
                fig = px.line(
                    daily_traffic,
                    x="timestamp",
                    y="average_speed_kmph",
                    markers=True,
                    title="Daily Average Recorded Speed",
                    labels={
                        "timestamp": "Date",
                        "average_speed_kmph": "Average Speed (km/h)"
                    }
                )
                fig.update_layout(height=370)
                st.plotly_chart(fig, use_container_width=True)

            if "road_name" in city_traffic.columns:
                road_data = city_traffic.copy()
                road_data["road_name"] = (
                    road_data["road_name"].fillna("Not specified").astype(str)
                )
                road_summary = (
                    road_data.groupby("road_name", as_index=False)
                    .agg(
                        vehicle_count=("vehicle_count", "sum"),
                        average_speed_kmph=("average_speed_kmph", "mean"),
                        observations=("timestamp", "count")
                    )
                    .sort_values("vehicle_count", ascending=False)
                )
                st.markdown("#### Road-wise Traffic Summary")
                st.dataframe(
                    road_summary.round(2),
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown("#### Uploaded Traffic Records")
            st.dataframe(
                city_traffic,
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "⬇️ Download Selected City Traffic Data",
                data=city_traffic.to_csv(index=False),
                file_name=f"{selected_city.lower()}_traffic_analysis.csv",
                mime="text/csv",
                key="traffic_city_download"
            )

            st.caption(
                "Traffic metrics are calculated only from the uploaded observations. "
                "Vehicle counts depend on the measurement interval and collection method; "
                "they are not automatically comparable across datasets."
            )

    st.divider()

    # --------------------------------------------------------
    # HISTORICAL JICA TRAFFIC SURVEYS (2011)
    # --------------------------------------------------------

    st.markdown("### 🚦 Historical JICA Traffic Survey (2011)")
    st.caption(
        "Historical field survey data from the JICA Hyderabad traffic study. "
        "These are 2011 survey totals, not live traffic observations or current conditions."
    )

    if selected_city != "Hyderabad":
        st.info(
            "JICA traffic survey records in this project are available for Hyderabad only. "
            "No Hyderabad traffic values are reused for another city."
        )
    elif jica_traffic.empty and jica_pcu.empty:
        st.warning(
            "JICA traffic files were not found or did not match the expected columns. "
            "Place both CSV files in data/raw/ to enable this section."
        )
    else:
        if not jica_traffic.empty:
            st.markdown("#### Directional and Movement Counts")
            traffic_view = jica_traffic.copy()
            traffic_view["location_name"] = traffic_view["location_name"].fillna("Not specified")
            traffic_view["direction_or_movement"] = traffic_view[
                "direction_or_movement"
            ].fillna("Not specified")

            count_types = sorted(traffic_view["count_type"].dropna().astype(str).unique())
            selected_count_type = st.selectbox(
                "Survey count type", count_types, key="jica_count_type"
            )
            filtered_traffic = traffic_view[
                traffic_view["count_type"].astype(str) == selected_count_type
            ].copy()

            locations = sorted(filtered_traffic["location_name"].astype(str).unique())
            selected_location = st.selectbox(
                "Survey location", ["All locations"] + locations,
                key="jica_location"
            )
            if selected_location != "All locations":
                filtered_traffic = filtered_traffic[
                    filtered_traffic["location_name"].astype(str) == selected_location
                ]

            m1, m2, m3 = st.columns(3)
            m1.metric("Survey Records", f"{len(filtered_traffic):,}")
            m2.metric("Locations", f"{filtered_traffic['location_name'].nunique():,}")
            duration = filtered_traffic["survey_duration_hours"].dropna()
            m3.metric(
                "Survey Duration",
                f"{duration.max():g} hours" if not duration.empty else "Not recorded"
            )

            if not filtered_traffic.empty:
                plot_data = (
                    filtered_traffic.groupby(
                        ["location_name", "direction_or_movement"], as_index=False
                    )["vehicle_count"].sum()
                )
                fig = px.bar(
                    plot_data,
                    x="location_name",
                    y="vehicle_count",
                    color="direction_or_movement",
                    barmode="group",
                    title="Reported Vehicle Counts by Location and Direction/Movement",
                    labels={
                        "location_name": "Survey Location",
                        "vehicle_count": "Reported Vehicle Count",
                        "direction_or_movement": "Direction / Movement"
                    }
                )
                fig.update_layout(height=430, xaxis_tickangle=-35)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                filtered_traffic,
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "⬇️ Download JICA Directional Traffic Records",
                data=filtered_traffic.to_csv(index=False).encode("utf-8"),
                file_name="hyderabad_jica_traffic_counts_2011_filtered.csv",
                mime="text/csv",
                key="jica_traffic_download"
            )
            st.caption(
                "Counts are reported by direction or movement. Do not add opposing "
                "directions and interpret the result as unique vehicles; the same "
                "traffic stream may be represented in separate survey rows."
            )
        else:
            st.info("Directional JICA traffic count file is not available.")

        st.divider()
        st.markdown("#### Passenger Car Unit (PCU) Survey Totals")
        if not jica_pcu.empty:
            pcu_view = jica_pcu.copy()
            c1, c2, c3 = st.columns(3)
            c1.metric("PCU Survey Locations", f"{len(pcu_view):,}")
            c2.metric("Largest Location Total", f"{pcu_view['vehicle_count_pcu'].max():,.0f} PCU")
            c3.metric("Survey Durations", f"{pcu_view['survey_duration_hours'].nunique()} types")

            fig = px.bar(
                pcu_view.sort_values("vehicle_count_pcu", ascending=False),
                x="location_name",
                y="vehicle_count_pcu",
                color="survey_duration_hours",
                title="JICA Reported PCU Totals by Survey Location",
                labels={
                    "location_name": "Survey Location",
                    "vehicle_count_pcu": "Reported PCU Total",
                    "survey_duration_hours": "Survey Duration (hours)"
                },
                hover_data=["survey_duration_hours", "derived_pcu_per_hour"]
            )
            fig.update_layout(height=430, xaxis_tickangle=-35)
            st.plotly_chart(fig, use_container_width=True)

            pcu_display = pcu_view[
                ["location_name", "survey_date", "survey_duration_hours",
                 "vehicle_count_pcu", "derived_pcu_per_hour"]
            ].copy()
            pcu_display = pcu_display.rename(columns={
                "location_name": "Survey Location",
                "survey_date": "Survey Date",
                "survey_duration_hours": "Survey Duration (hours)",
                "vehicle_count_pcu": "Reported PCU Total",
                "derived_pcu_per_hour": "Derived Average PCU/hour"
            })
            st.dataframe(
                pcu_display.round({"Derived Average PCU/hour": 2}),
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "⬇️ Download JICA PCU Survey Records",
                data=pcu_view.to_csv(index=False).encode("utf-8"),
                file_name="hyderabad_jica_traffic_pcu_2011.csv",
                mime="text/csv",
                key="jica_pcu_download"
            )
            st.caption(
                "PCU totals are passenger-car-unit equivalents, not raw vehicle counts. "
                "Derived average PCU/hour is calculated by dividing the reported total "
                "by the survey duration; it is not an observed hourly time series. "
                "Any 07:00 timestamp in the supplied PCU CSV is treated only as a "
                "date placeholder and is not displayed as an observation time."
            )
        else:
            st.info("JICA PCU survey file is not available or has an unsupported format.")

    st.divider()

    # --------------------------------------------------------
    # HISTORICAL WEATHER ANALYSIS
    # --------------------------------------------------------

    st.markdown("### 🌦️ Historical Weather Analysis")

    if selected_city != "Hyderabad":
        st.info(
            "Historical weather data in the current project is available for "
            "Hyderabad only. No Hyderabad weather values are reused for other cities."
        )
    elif weather.empty:
        st.error("Hyderabad weather dataset not found.")
    elif "time" not in weather.columns:
        st.error("Weather dataset does not contain a time column.")
    else:
        data = weather.copy()
        data["time"] = pd.to_datetime(data["time"], errors="coerce")
        data = data.dropna(subset=["time"]).copy()
        data["year"] = data["time"].dt.year
        data["month"] = data["time"].dt.month

        numeric_columns = [
            col for col in data.columns
            if col not in ["time", "year", "month"]
            and pd.api.types.is_numeric_dtype(data[col])
        ]

        if data.empty:
            st.warning("No valid dated weather records were found.")
        elif not numeric_columns:
            st.warning("No numeric weather variables found.")
        else:
            years = sorted(data["year"].dropna().unique().tolist())
            selected_year = st.selectbox(
                "Select Weather Year",
                years,
                key="weather_year_select"
            )
            year_data = data[data["year"] == selected_year].copy()

            variable_labels = {
                "temperature_2m": "Temperature (°C)",
                "relative_humidity_2m": "Relative Humidity (%)",
                "precipitation": "Precipitation (mm)",
                "rain": "Rainfall (mm)",
                "weather_code": "Weather Code"
            }
            selected_variable = st.selectbox(
                "Weather Variable",
                numeric_columns,
                format_func=lambda x: variable_labels.get(
                    x, x.replace("_", " ").title()
                ),
                key="weather_variable_select"
            )
            selected_label = variable_labels.get(
                selected_variable,
                selected_variable.replace("_", " ").title()
            )

            c1, c2, c3 = st.columns(3)
            c1.metric("Weather Records", f"{len(year_data):,}")
            c2.metric(
                "Average",
                f"{year_data[selected_variable].mean():.2f}"
            )
            c3.metric(
                "Maximum",
                f"{year_data[selected_variable].max():.2f}"
            )

            fig = px.line(
                year_data,
                x="time",
                y=selected_variable,
                title=f"{selected_label} - {selected_year}"
            )
            fig.update_layout(height=380)
            st.plotly_chart(fig, use_container_width=True)

            monthly = (
                year_data.groupby("month")[selected_variable]
                .mean()
                .reindex(range(1, 13))
                .dropna()
                .reset_index()
            )
            month_names = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
            }
            monthly["Month"] = monthly["month"].map(month_names)

            fig = px.bar(
                monthly,
                x="Month",
                y=selected_variable,
                title="Monthly Weather Pattern"
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("#### Monthly Weather Summary")
            st.dataframe(
                monthly[["Month", selected_variable]].round(2),
                use_container_width=True,
                hide_index=True
            )
            st.download_button(
                "⬇️ Download Selected Year Weather Data",
                data=year_data.to_csv(index=False),
                file_name=f"hyderabad_weather_{selected_year}.csv",
                mime="text/csv",
                key="weather_download"
            )
            st.caption(
                "Historical weather from the Open-Meteo Archive at a Hyderabad "
                "reference coordinate. It is not measured at individual accident "
                "locations and does not establish accident causation."
            )


# ============================================================
# 6. REPORTS & INSIGHTS
# ============================================================

elif page == "Reports & Insights":

    st.markdown(
        '<div class="section-title">📑 Reports & Project Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown("### City Road Data Availability")

    rows = []

    for city in CITIES:
        summary = ALL_SUMMARIES[city]

        if summary.empty:
            rows.append(
                {
                    "City": city,
                    "Road Data": "Not available",
                    "Road Segments": 0,
                    "Mapped Length (km)": 0
                }
            )
        else:
            count, length, _ = summary_metrics(summary)

            rows.append(
                {
                    "City": city,
                    "Road Data": "Available",
                    "Road Segments": count,
                    "Mapped Length (km)": round(length, 2)
                }
            )

    availability = pd.DataFrame(rows)

    st.dataframe(
        availability,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇️ Download Six-City Road Availability",
        data=availability.to_csv(index=False),
        file_name="smartroad_six_city_road_availability.csv",
        mime="text/csv"
    )

    st.divider()

    st.markdown(
        f"### {selected_city} Road Summary"
    )

    summary = selected_summary

    if not summary.empty:

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "⬇️ Download Selected City Road Summary",
            data=summary.to_csv(index=False),
            file_name=(
                f"{selected_city.lower()}_road_summary.csv"
            ),
            mime="text/csv"
        )

    else:
        st.warning("No road summary available for this city.")

    st.divider()

    st.markdown("### Hyderabad Accident and Weather Datasets")

    if not accidents.empty:
        st.download_button(
            "Download Hyderabad Accident Dataset",
            data=accidents.to_csv(index=False),
            file_name="hyderabad_accidents.csv",
            mime="text/csv"
        )

    if not weather.empty:
        st.download_button(
            "Download Hyderabad Weather Dataset",
            data=weather.to_csv(index=False),
            file_name="hyderabad_weather.csv",
            mime="text/csv"
        )

    st.divider()

    st.markdown("### Project Data Sources")

    st.markdown("""
    1. **OpenStreetMap:** Road network features.
    2. **BBBike Extract:** Downloaded city road datasets.
    3. **Open-Meteo Archive:** Historical Hyderabad weather.
    4. **Hyderabad Accident Dataset:** Annual accident counts.

    Accident and weather data are currently available only for Hyderabad.
    Road network availability is shown separately for each city.
    """)

    st.warning(
        "The application does not claim live traffic monitoring, "
        "verified accident hotspots, or validated accident prediction "
        "for cities without supporting datasets."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "🚦 SmartRoad AI | Smart City Road Safety & Infrastructure "
    "Intelligence | MCA Data Analytics & Visualization Project"
)
