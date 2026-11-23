
import json
import geopandas as gpd
import pandas as pd

from pathlib import Path
from shapely.geometry import Point
from city_config import CITIES


# ==========================================
# SMARTROAD AI - MULTI-CITY ROAD PROCESSING
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_CITIES_DIR = BASE_DIR / "data" / "raw" / "cities"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Common study area for all cities
STUDY_RADIUS_KM = 8

# Planned roads are stored separately
PLANNED_TYPES = {"proposed", "construction"}


# ==========================================
# FIND AVAILABLE CITY DATASETS
# ==========================================

def find_city_files():

    available_cities = {}

    if not RAW_CITIES_DIR.exists():
        print("Raw cities folder does not exist.")
        return available_cities

    for city_folder in sorted(RAW_CITIES_DIR.iterdir()):

        if not city_folder.is_dir():
            continue

        geojson_files = list(city_folder.glob("*.osm.geojson"))

        if geojson_files:
            available_cities[city_folder.name] = geojson_files[0]

    return available_cities


# ==========================================
# EXTRACT HIGHWAY CLASSIFICATION
# ==========================================

def get_highway(row):

    highway = row.get("highway")

    if isinstance(highway, str) and highway.strip():
        return highway

    if isinstance(highway, list):
        return ";".join(str(item) for item in highway)

    tags = row.get("tags")

    if isinstance(tags, str):
        try:
            tags = json.loads(tags)
        except (json.JSONDecodeError, TypeError):
            tags = {}

    if isinstance(tags, dict):

        highway = tags.get("highway")

        if isinstance(highway, list):
            return ";".join(str(item) for item in highway)

        if highway:
            return str(highway)

    return None


# ==========================================
# PROCESS ONE CITY
# ==========================================

def process_city(city, input_file):

    print("\n==========================================")
    print(f" PROCESSING: {city.upper()}")
    print("==========================================")

    output_folder = PROCESSED_DIR / city
    output_folder.mkdir(parents=True, exist_ok=True)

    try:

        # ----------------------------------
        # CITY CONFIGURATION
        # ----------------------------------

        city_name = city.title()

        if city_name not in CITIES:
            print("City configuration not found:", city_name)
            return False

        config = CITIES[city_name]

        latitude = config["latitude"]
        longitude = config["longitude"]

        print("Loading GeoJSON:", input_file)

        # ----------------------------------
        # LOAD RAW DATA
        # ----------------------------------

        data = gpd.read_file(input_file)

        print("Total source features:", len(data))

        if data.empty:
            print("Dataset is empty.")
            return False

        if data.crs is None:
            print("ERROR: Input dataset has no CRS.")
            return False

        # ----------------------------------
        # IDENTIFY ROAD TYPES
        # ----------------------------------

        print("Identifying road classifications...")

        data["road_type"] = data.apply(
            get_highway,
            axis=1
        )

        roads = data[
            data["road_type"].notna()
            & data.geometry.notna()
            & ~data.geometry.is_empty
            & data.geometry.geom_type.isin(
                ["LineString", "MultiLineString"]
            )
        ].copy()

        if roads.empty:
            print("No road features found.")
            return False

        print("Road features identified:", len(roads))

        # ----------------------------------
        # CREATE 8 KM STUDY AREA
        # ----------------------------------

        centre = gpd.GeoSeries(
            [Point(longitude, latitude)],
            crs="EPSG:4326"
        )

        projected_crs = centre.estimate_utm_crs()

        if projected_crs is None:
            print("Could not estimate projected CRS.")
            return False

        print("Using projected CRS:", projected_crs)

        # Project road geometries
        roads = roads.to_crs(projected_crs)

        # Project city centre
        centre_projected = centre.to_crs(projected_crs)

        # Create circular boundary
        study_area = centre_projected.iloc[0].buffer(
            STUDY_RADIUS_KM * 1000
        )

        print(
            f"Clipping roads to {STUDY_RADIUS_KM} km radius..."
        )

        # ----------------------------------
        # CLIP ROADS TO STUDY AREA
        # ----------------------------------

        roads["geometry"] = roads.geometry.intersection(
            study_area
        )

        roads = roads[
            roads.geometry.notna()
            & ~roads.geometry.is_empty
            & roads.geometry.geom_type.isin(
                ["LineString", "MultiLineString"]
            )
        ].copy()

        if roads.empty:
            print("No roads inside the study area.")
            return False

        # ----------------------------------
        # CALCULATE ROAD LENGTH
        # ----------------------------------

        roads["length_m"] = roads.geometry.length

        roads["length_km"] = (
            roads["length_m"] / 1000
        )

        # ----------------------------------
        # SEPARATE PLANNED AND OPERATIONAL
        # ----------------------------------

        road_types = (
            roads["road_type"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        planned_mask = road_types.isin(
            PLANNED_TYPES
        )

        planned_roads = roads[planned_mask].copy()

        operational_roads = roads[
            ~planned_mask
        ].copy()

        # ----------------------------------
        # CONVERT BACK TO GEOGRAPHIC CRS
        # ----------------------------------

        operational_roads = operational_roads.to_crs(
            "EPSG:4326"
        )

        planned_roads = planned_roads.to_crs(
            "EPSG:4326"
        )

        # ----------------------------------
        # OPERATIONAL ROAD SUMMARY
        # ----------------------------------

        road_summary = (
            operational_roads.groupby("road_type")
            .agg(
                road_segments=("road_type", "size"),
                total_length_km=("length_km", "sum")
            )
            .reset_index()
            .sort_values(
                "road_segments",
                ascending=False
            )
        )

        # ----------------------------------
        # PLANNED ROAD SUMMARY
        # ----------------------------------

        if not planned_roads.empty:

            planned_summary = (
                planned_roads.groupby("road_type")
                .agg(
                    road_segments=("road_type", "size"),
                    total_length_km=("length_km", "sum")
                )
                .reset_index()
            )

        else:

            planned_summary = pd.DataFrame(
                columns=[
                    "road_type",
                    "road_segments",
                    "total_length_km"
                ]
            )

        # ----------------------------------
        # SAVE OPERATIONAL ROAD DATA
        # ----------------------------------

        geojson_path = (
            output_folder / f"{city}_roads.geojson"
        )

        csv_path = (
            output_folder / f"{city}_roads.csv"
        )

        summary_path = (
            output_folder / "road_type_summary.csv"
        )

        operational_roads.to_file(
            geojson_path,
            driver="GeoJSON"
        )

        operational_roads.drop(
            columns="geometry"
        ).to_csv(
            csv_path,
            index=False
        )

        road_summary.to_csv(
            summary_path,
            index=False
        )

        # ----------------------------------
        # SAVE PLANNED ROAD DATA
        # ----------------------------------

        planned_geojson_path = (
            output_folder / f"{city}_planned_roads.geojson"
        )

        planned_summary_path = (
            output_folder / "planned_road_summary.csv"
        )

        planned_roads.to_file(
            planned_geojson_path,
            driver="GeoJSON"
        )

        planned_summary.to_csv(
            planned_summary_path,
            index=False
        )

        # ----------------------------------
        # DISPLAY RESULTS
        # ----------------------------------

        operational_length = operational_roads[
            "length_km"
        ].sum()

        planned_length = planned_roads[
            "length_km"
        ].sum()

        print("\n========== PROCESSING SUCCESS ==========")

        print("City:", city_name)

        print("Study radius:", STUDY_RADIUS_KM, "km")

        print(
            "Operational road segments:",
            len(operational_roads)
        )

        print(
            "Operational mapped network length:",
            round(operational_length, 2),
            "km"
        )

        print(
            "Planned road segments:",
            len(planned_roads)
        )

        print(
            "Planned road length:",
            round(planned_length, 2),
            "km"
        )

        print("\nOperational road summary:")

        print(
            road_summary.to_string(index=False)
        )

        print("\nSaved files:")

        print(geojson_path)
        print(csv_path)
        print(summary_path)
        print(planned_geojson_path)
        print(planned_summary_path)

        print("\nProcessing completed successfully!")

        return True

    except Exception as error:

        print("\nPROCESSING FAILED")

        print("City:", city)

        print("Error:", error)

        return False


# ==========================================
# PROCESS ALL SIX CITIES
# ==========================================

if __name__ == "__main__":

    print("\n==========================================")
    print(" SMARTROAD AI - MULTI-CITY PROCESSING")
    print("==========================================")

    print("Study radius:", STUDY_RADIUS_KM, "km")

    available_cities = find_city_files()

    if not available_cities:

        print("\nNo city datasets found.")

    else:

        print(
            "\nAvailable cities:",
            list(available_cities.keys())
        )

        results = {}

        for city, input_file in available_cities.items():

            results[city] = process_city(
                city,
                input_file
            )

        # ----------------------------------
        # FINAL PROCESSING SUMMARY
        # ----------------------------------

        print("\n==========================================")
        print(" MULTI-CITY PROCESSING SUMMARY")
        print("==========================================")

        for city, success in results.items():

            status = (
                "SUCCESS" if success else "FAILED"
            )

            print(f"{city.title():15} : {status}")

        successful = sum(results.values())

        print("\nTotal cities:", len(results))

        print("Successful:", successful)

        print("Failed:", len(results) - successful)

        print("\nProcessing finished.")
