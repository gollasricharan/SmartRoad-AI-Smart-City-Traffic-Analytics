
import osmnx as ox
from pathlib import Path

print("Starting Hyderabad road network download...")

place = "Hyderabad, Telangana, India"

# OSMnx settings
ox.settings.use_cache = True
ox.settings.log_console = True
ox.settings.overpass_url = "https://overpass.kumi.systems/api"
ox.settings.requests_timeout = 600

# Create output folder
output = Path("data/raw")
output.mkdir(parents=True, exist_ok=True)

try:
    print("\nDownloading Hyderabad road network...")
    print("Please wait. This may take several minutes.")

    # Download driving road network
    G = ox.graph_from_place(
        place,
        network_type="drive",
        simplify=True
    )

    print("\nRoad network downloaded successfully!")

    # Convert graph into GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(G)

    # Save GraphML
    ox.save_graphml(
        G,
        filepath=output / "hyderabad_roads.graphml"
    )

    # Save CSV
    roads = edges.reset_index()

    roads.to_csv(
        output / "hyderabad_roads.csv",
        index=False
    )

    # Save GeoJSON
    roads.to_file(
        output / "hyderabad_roads.geojson",
        driver="GeoJSON"
    )

    # Download Hyderabad boundary
    print("\nDownloading Hyderabad boundary...")

    boundary = ox.geocode_to_gdf(place)

    boundary.to_file(
        output / "hyderabad_boundary.geojson",
        driver="GeoJSON"
    )

    # Display summary
    print("\n========== ROAD NETWORK SUMMARY ==========")
    print("Total road segments:", len(edges))
    print("Total intersections/nodes:", len(nodes))

    print("\nTop 10 road types:")
    print(edges["highway"].value_counts().head(10))

    print("\n========== SAVED FILES ==========")
    print("1. hyderabad_roads.graphml")
    print("2. hyderabad_roads.csv")
    print("3. hyderabad_roads.geojson")
    print("4. hyderabad_boundary.geojson")

    print("\nDownload completed successfully!")

except Exception as error:
    print("\nROAD NETWORK DOWNLOAD FAILED")
    print("Error:", error)
    print("\nShare the complete terminal output.")
