
import osmnx as ox
from pathlib import Path
from src.city_config import CITIES
import time

print("\n====================================")
print(" SMARTROAD AI - ROAD NETWORK")
print("====================================")

# ==========================================
# OSMNX SETTINGS
# ==========================================

ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.overpass_url = "https://overpass.private.coffee/api"
ox.settings.timeout = 240
ox.settings.requests_timeout = 240

# ==========================================
# DISPLAY CITIES
# ==========================================

city_names = list(CITIES.keys())

print("\nAvailable Cities:\n")

for i, city in enumerate(city_names, start=1):
    print(f"{i}. {city}")

# ==========================================
# CITY SELECTION
# ==========================================

try:
    choice = int(input("\nSelect city number (0 to exit): "))

    if choice == 0:
        print("Download cancelled.")
        raise SystemExit

    if choice < 1 or choice > len(city_names):
        print("Invalid selection.")
        raise SystemExit

except ValueError:
    print("Please enter a valid number.")
    raise SystemExit

selected_city = city_names[choice - 1]
details = CITIES[selected_city]

latitude = details["latitude"]
longitude = details["longitude"]

# ==========================================
# OUTPUT DIRECTORY
# ==========================================

radius = 3000

output = (
    Path("data/raw/cities")
    / selected_city.lower().replace(" ", "_")
)

output.mkdir(parents=True, exist_ok=True)

print("\nSelected city:", selected_city)
print("Latitude:", latitude)
print("Longitude:", longitude)
print("Area radius:", radius / 1000, "km")

# ==========================================
# CHECK EXISTING DATA
# ==========================================

if (
    (output / "roads.graphml").exists()
    and (output / "roads.geojson").exists()
):
    print("\nRoad network already exists.")
    print("Skipping download.")
    raise SystemExit

# ==========================================
# DOWNLOAD WITH RETRIES
# ==========================================

G = None

for attempt in range(1, 4):

    print(f"\nDownload attempt {attempt}/3")

    try:
        G = ox.graph_from_point(
            center_point=(latitude, longitude),
            dist=radius,
            dist_type="bbox",
            network_type="drive",
            simplify=True,
            retain_all=True
        )

        if len(G.nodes) == 0:
            raise ValueError("No road network data returned.")

        print("\nRoad network downloaded successfully!")
        break

    except Exception as error:
        print("\nDownload attempt failed:", error)

        if attempt < 3:
            wait_time = 15 * (2 ** (attempt - 1))
            print(f"Waiting {wait_time} seconds before retry...")
            time.sleep(wait_time)
        else:
            print("\nAll download attempts failed.")
            print("Try again later or use an OSM extract.")
            raise SystemExit

# ==========================================
# CONVERT GRAPH TO GEODATAFRAMES
# ==========================================

try:
    nodes, edges = ox.graph_to_gdfs(G)

    # Save GraphML
    ox.save_graphml(
        G,
        filepath=output / "roads.graphml"
    )

    # Save road segments
    roads = edges.reset_index()

    roads.to_csv(
        output / "roads.csv",
        index=False
    )

    roads.to_file(
        output / "roads.geojson",
        driver="GeoJSON"
    )

    # Save road nodes / intersections
    intersections = nodes.reset_index()

    intersections.to_file(
        output / "intersections.geojson",
        driver="GeoJSON"
    )

    # ======================================
    # SUCCESS SUMMARY
    # ======================================

    print("\n====================================")
    print(" ROAD DOWNLOAD SUCCESS")
    print("====================================")

    print("City:", selected_city)
    print("Road segments:", len(edges))
    print("Road nodes:", len(nodes))

    print("\nSaved files:")

    print(output / "roads.graphml")
    print(output / "roads.csv")
    print(output / "roads.geojson")
    print(output / "intersections.geojson")

    print("\nSMARTROAD AI download completed!")

except Exception as error:
    print("\n========== FILE SAVING FAILED ==========")
    print("Error:", error)
