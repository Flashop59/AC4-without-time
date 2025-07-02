import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union
from geopy.distance import geodesic
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import io

st.set_page_config(page_title="Field Analyzer", layout="wide")

def extract_coordinates(df):
    df = df.copy()

    if df.shape[1] < 6:
        raise ValueError("Expected at least 6 columns with GPS coordinates in column F.")

    # Clean and parse Column F safely
    gps_raw = df.iloc[:, 5].astype(str).str.strip()

    lat_list = []
    lon_list = []

    for value in gps_raw:
        if "," not in value:
            continue
        parts = value.split(",")
        if len(parts) != 2:
            continue
        try:
            lat = float(parts[0])
            lon = float(parts[1])
            lat_list.append(lat)
            lon_list.append(lon)
        except:
            continue  # Skip bad rows

    if not lat_list or not lon_list:
        raise ValueError("No usable GPS data found after filtering.")

    return pd.DataFrame({"latitude": lat_list, "longitude": lon_list})

def haversine_area(poly):
    import pyproj
    from shapely.ops import transform
    from functools import partial

    proj = partial(
        pyproj.transform,
        pyproj.Proj(init='epsg:4326'),
        pyproj.Proj(proj='aea', lat_1=poly.bounds[1], lat_2=poly.bounds[3])
    )
    poly_area = transform(proj, poly).area
    return poly_area

def calculate_concave_hull(points, alpha=0.01):
    try:
        import alphashape
        return alphashape.alphashape(points, alpha)
    except:
        return MultiPoint(points).convex_hull

def process_data(df, show_hull=True):
    coords = extract_coordinates(df)
    db = DBSCAN(eps=0.0005, min_samples=5).fit(coords)
    coords["field_id"] = db.labels_

    result = []
    field_polygons = []
    centroids = []

    for field_id in sorted(coords["field_id"].unique()):
        if field_id == -1:
            continue
        field_points = coords[coords["field_id"] == field_id][["latitude", "longitude"]].values
        polygon = calculate_concave_hull(field_points, alpha=0.01)

        if not isinstance(polygon, Polygon):
            continue

        area_m2 = haversine_area(polygon)
        area_guntha = area_m2 / 101.17
        if area_guntha < 5:
            continue

        centroid = polygon.centroid
        result.append({
            "Field ID": field_id,
            "Area (Gunthas)": round(area_guntha, 2),
        })
        centroids.append((centroid.y, centroid.x))
        field_polygons.append((field_id, polygon))

    travel_distances = []
    for i in range(len(centroids) - 1):
        travel_distances.append(round(geodesic(centroids[i], centroids[i + 1]).km, 2))
    travel_distances.append("-")

    for i, dist in enumerate(travel_distances):
        result[i]["Travel Distance to Next Field (km)"] = dist

    return result, field_polygons, coords

def render_map(coords, field_polygons, show_hull):
    if coords.empty:
        return None

    map_center = [coords["latitude"].mean(), coords["longitude"].mean()]
    fmap = folium.Map(location=map_center, zoom_start=17, tiles='CartoDB Positron')
    folium.TileLayer("Esri.WorldImagery").add_to(fmap)

    marker_cluster = MarkerCluster().add_to(fmap)

    for idx, row in coords.iterrows():
        folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=2,
            fill=True,
            color="blue" if row["field_id"] != -1 else "gray",
            fill_opacity=0.7
        ).add_to(marker_cluster)

    if show_hull:
        for field_id, poly in field_polygons:
            if not isinstance(poly, Polygon):
                continue
            folium.Polygon(
                locations=[(lat, lon) for lon, lat in poly.exterior.coords],
                tooltip=f"Field {field_id}",
                color="green",
                fill=True,
                fill_opacity=0.3
            ).add_to(fmap)

    return fmap

# --- Streamlit Interface ---
st.title("🚜 Field Analyzer – Area & Travel Distance Calculator")
uploaded_file = st.file_uploader("📂 Upload GPS Report CSV", type="csv")
show_hull = st.checkbox("Show Field Boundaries (Concave Hulls)", value=True)

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        summary, field_polygons, coords = process_data(df, show_hull)
        st.success("✅ Data Processed Successfully.")

        # Summary Table
        summary_df = pd.DataFrame(summary)
        st.subheader("📊 Field Summary")
        st.dataframe(summary_df)

        # Download Button
        csv_buf = io.StringIO()
        summary_df.to_csv(csv_buf, index=False)
        st.download_button("📥 Download Summary as CSV", csv_buf.getvalue(), file_name="field_summary.csv")

        # Map Visualization
        st.subheader("🗺️ Field Map")
        fmap = render_map(coords, field_polygons, show_hull)
        if fmap:
            st_folium(fmap, width=900, height=600)
        else:
            st.warning("Map could not be rendered.")

    except Exception as e:
        st.error(f"❌ Error processing file:\n\n{e}")
