import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import MultiPoint
from shapely.ops import unary_union, polygonize
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
from geopy.distance import geodesic
import folium
from folium import plugins
from streamlit_folium import folium_static


# ------------------------ CONCAVE HULL ------------------------
def alpha_shape(points, alpha=0.02):
    if len(points) < 4:
        return MultiPoint(list(points)).convex_hull

    tri = Delaunay(points)
    triangles = points[tri.simplices]

    a = np.linalg.norm(triangles[:, 0] - triangles[:, 1], axis=1)
    b = np.linalg.norm(triangles[:, 1] - triangles[:, 2], axis=1)
    c = np.linalg.norm(triangles[:, 2] - triangles[:, 0], axis=1)
    s = (a + b + c) / 2.0
    area = np.sqrt(s * (s - a) * (s - b) * (s - c))
    circum_r = a * b * c / (4.0 * area)
    keep = circum_r < 1.0 / alpha

    edges = set()

    def add_edge(i, j):
        if (i, j) in edges or (j, i) in edges:
            edges.discard((j, i))
        else:
            edges.add((i, j))

    for simplex in tri.simplices[keep]:
        add_edge(simplex[0], simplex[1])
        add_edge(simplex[1], simplex[2])
        add_edge(simplex[2], simplex[0])

    edge_lines = [(points[i], points[j]) for i, j in edges]
    polygons = list(polygonize(edge_lines))
    return unary_union(polygons)


def calculate_concave_hull_area(points):
    try:
        hull = alpha_shape(points)
        return hull.area
    except Exception:
        return 0.0


def calculate_centroid(points):
    return np.mean(points, axis=0)


def generate_more_hull_points(coords, num_splits=3):
    new_points = []
    for i in range(len(coords)):
        start = coords[i]
        end = coords[(i + 1) % len(coords)]
        new_points.append(start)
        for j in range(1, num_splits):
            interp = start + (end - start) * j / num_splits
            new_points.append(interp)
    return np.array(new_points)


# ------------------------ PROCESSING ------------------------
def process_gps_data(df, show_hull):
    coords_split = df["Address"].str.split(",", expand=True)
    df["lat"] = coords_split[0].astype(str).str.strip()
    df["lng"] = coords_split[1].astype(str).str.strip()

    # Convert to float safely
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lng"] = pd.to_numeric(df["lng"], errors="coerce")

    # Drop rows with invalid coordinates
    df = df.dropna(subset=["lat", "lng"])

    coords = df[["lat", "lng"]].values

    # Cluster points
    db = DBSCAN(eps=0.00003, min_samples=12)
    df["field_id"] = db.fit_predict(coords)

    clustered = df[df["field_id"] != -1]

    if clustered.empty:
        return None, None, 0, 0

    # Area calculation
    field_areas_raw = clustered.groupby("field_id").apply(
        lambda g: calculate_concave_hull_area(g[["lat", "lng"]].values)
    )
    field_areas_m2 = field_areas_raw * 0.77 * (111000 ** 2)
    field_areas_gunthas = field_areas_m2 / 101.17

    # Filter small fields
    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index
    field_areas_gunthas = field_areas_gunthas.loc[valid_fields]

    if field_areas_gunthas.empty:
        return None, None, 0, 0

    # Centroids
    centroids = clustered.groupby("field_id").apply(
        lambda g: calculate_centroid(g[["lat", "lng"]].values)
    )

    # Travel distances
    travel_distances = []
    field_ids = list(valid_fields)

    if len(field_ids) > 1:
        for i in range(len(field_ids) - 1):
            c1 = centroids.loc[field_ids[i]]
            c2 = centroids.loc[field_ids[i + 1]]
            dist = geodesic(c1, c2).kilometers
            travel_distances.append(dist)
        travel_distances.append(np.nan)
    else:
        travel_distances.append(np.nan)

    # Build summary DataFrame
    summary_df = pd.DataFrame({
        "Field ID": field_areas_gunthas.index,
        "Area (Gunthas)": field_areas_gunthas.values,
        "Travel Distance to Next Field (km)": travel_distances
    })

    total_area = field_areas_gunthas.sum()
    total_travel_distance = np.nansum(travel_distances)

    # Map
    map_center = [df["lat"].mean(), df["lng"].mean()]
    fmap = folium.Map(location=map_center, zoom_start=13)

    folium.TileLayer(
        tiles="https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbHo5NzkycmIwN2RxMmtzZHZvNWpjYmQ2In0.A_FZYl5zKjwSZpJuP_MHiA",
        attr="Mapbox Satellite",
        name="Satellite",
        overlay=True,
        control=True
    ).add_to(fmap)

    plugins.Fullscreen().add_to(fmap)

    for _, row in df.iterrows():
        color = "blue" if row["field_id"] in valid_fields else "red"
        folium.CircleMarker(
            location=[row["lat"], row["lng"]],
            radius=2,
            color=color,
            fill=True
        ).add_to(fmap)

    if show_hull:
        for fid in valid_fields:
            pts = clustered[clustered["field_id"] == fid][["lat", "lng"]].values
            hull = alpha_shape(pts)
            if hull.geom_type == "Polygon":
                coords = np.array(hull.exterior.coords)
                folium.Polygon(
                    locations=coords.tolist(),
                    color="green",
                    fill=True,
                    fill_opacity=0.5
                ).add_to(fmap)

                dense_pts = generate_more_hull_points(coords)
                folium.PolyLine(
                    locations=dense_pts.tolist(),
                    color="yellow",
                    weight=2
                ).add_to(fmap)

    return fmap, summary_df, total_area, total_travel_distance


# ------------------------ STREAMLIT APP ------------------------
def main():
    st.set_page_config(page_title="Field Analyzer", layout="wide")
    st.title("Field CSV Analyzer (Lat/Lon from Address)")

    uploaded = st.file_uploader("Upload CSV with 'Address' column containing 'lat,lon'", type=["csv"])
    show_hull = st.checkbox("Show Concave Hulls", value=True)

    if uploaded:
        df = pd.read_csv(uploaded)

        if "Address" not in df.columns:
            st.error("CSV must have an 'Address' column with lat,lon format.")
            return

        result = process_gps_data(df, show_hull)

        if result[0] is None:
            st.warning("No valid clusters detected.")
            return

        fmap, summary_df, total_area, total_travel_distance = result

        st.success("Analysis completed.")
        st.subheader("Field Summary")
        st.dataframe(summary_df)

        st.subheader("Totals")
        st.markdown(f"**Total Area:** {total_area:.2f} gunthas")
        st.markdown(f"**Total Travel Distance:** {total_travel_distance:.2f} km")

        st.subheader("Map Visualization")
        folium_static(fmap)


if __name__ == "__main__":
    main()
