import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Polygon, MultiPoint
from shapely.ops import unary_union, polygonize
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay
import folium
from folium import plugins
from geopy.distance import geodesic
from streamlit_folium import folium_static

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

    for i, triangle in enumerate(tri.simplices[keep]):
        add_edge(triangle[0], triangle[1])
        add_edge(triangle[1], triangle[2])
        add_edge(triangle[2], triangle[0])

    m = MultiPoint(points)
    edge_lines = [(points[i], points[j]) for i, j in edges]
    polygon = list(polygonize(edge_lines))
    return unary_union(polygon)

def calculate_concave_hull_area(points):
    try:
        shape = alpha_shape(points)
        return shape.area
    except Exception:
        return 0

def calculate_centroid(points):
    return np.mean(points, axis=0)

def generate_more_hull_points(coords, num_splits=3):
    new_points = []
    for i in range(len(coords)):
        start = coords[i]
        end = coords[(i + 1) % len(coords)]
        new_points.append(start)
        for j in range(1, num_splits):
            point = start + (end - start) * j / num_splits
            new_points.append(point)
    return np.array(new_points)

def process_csv_data(gps_data, show_hull_points):
    gps_data['lat'] = gps_data['lat'].astype(float)
    gps_data['lng'] = gps_data['lon'].astype(float)

    coords = gps_data[['lat', 'lng']].values
    db = DBSCAN(eps=0.0001, min_samples=12).fit(coords)
    gps_data['field_id'] = db.labels_

    fields = gps_data[gps_data['field_id'] != -1]

    field_areas = fields.groupby('field_id').apply(
        lambda df: calculate_concave_hull_area(df[['lat', 'lng']].values))
    field_areas_m2 = field_areas * 0.77 * (111000 ** 2)
    field_areas_gunthas = field_areas_m2 / 101.17

    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index
    field_areas_gunthas = field_areas_gunthas[valid_fields]

    centroids = fields.groupby('field_id').apply(lambda df: calculate_centroid(df[['lat', 'lng']].values))

    travel_distances = []
    field_ids = list(valid_fields)

    if len(field_ids) > 1:
        for i in range(len(field_ids) - 1):
            c1 = centroids.loc[field_ids[i]]
            c2 = centroids.loc[field_ids[i + 1]]
            distance = geodesic(c1, c2).kilometers
            travel_distances.append(distance)
        travel_distances.append(np.nan)  # Last one has no next field
    else:
        travel_distances.append(np.nan)

    if len(travel_distances) != len(field_areas_gunthas):
        travel_distances = travel_distances[:len(field_areas_gunthas)]

    combined_df = pd.DataFrame({
        'Field ID': field_areas_gunthas.index,
        'Area (Gunthas)': field_areas_gunthas.values,
        'Travel Distance to Next Field (km)': travel_distances
    })

    total_area = field_areas_gunthas.sum()
    total_travel_distance = np.nansum(travel_distances)

    map_center = [gps_data['lat'].mean(), gps_data['lng'].mean()]
    m = folium.Map(location=map_center, zoom_start=12)

    folium.TileLayer(
        tiles='https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{z}/{x}/{y}?access_token=pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbHo5NzkycmIwN2RxMmtzZHZvNWpjYmQ2In0.A_FZYl5zKjwSZpJuP_MHiA',
        attr='Mapbox Satellite',
        name='Satellite',
        overlay=True,
        control=True
    ).add_to(m)
    plugins.Fullscreen().add_to(m)

    for _, row in gps_data.iterrows():
        color = 'blue' if row['field_id'] in valid_fields else 'red'
        folium.CircleMarker(location=[row['lat'], row['lng']], radius=2, color=color, fill=True).add_to(m)

    if show_hull_points:
        for field_id in valid_fields:
            points = fields[fields['field_id'] == field_id][['lat', 'lng']].values
            hull_shape = alpha_shape(points)
            if hull_shape.geom_type == 'Polygon':
                coords = np.array(hull_shape.exterior.coords)
                folium.Polygon(locations=coords.tolist(), color='green', fill=True, fill_opacity=0.5).add_to(m)
                more_pts = generate_more_hull_points(coords)
                folium.PolyLine(locations=more_pts.tolist(), color='yellow', weight=2).add_to(m)

    return m, combined_df, total_area, total_travel_distance

def main():
    st.title("Field Area & Travel Analyzer (No Time Required)")

    uploaded_file = st.file_uploader("Upload CSV file (must contain 'lat' and 'lon')", type=["csv"])
    show_hull_points = st.checkbox("Show Hull Points", value=False)

    if uploaded_file:
        gps_data = pd.read_csv(uploaded_file)

        required_columns = {'lat', 'lon'}
        if not required_columns.issubset(gps_data.columns):
            st.error("CSV must contain columns: lat, lon")
            return

        m, df, total_area, total_travel_dist = process_csv_data(gps_data, show_hull_points)

        st.success("Analysis Completed.")
        st.subheader("Field Data Summary")
        st.dataframe(df)

        st.subheader("Total Metrics")
        st.markdown(f"**Total Area:** {total_area:.2f} gunthas")
        st.markdown(f"**Total Travel Distance:** {total_travel_dist:.2f} km")

        st.subheader("Field Map")
        folium_static(m)

if __name__ == "__main__":
    main()
