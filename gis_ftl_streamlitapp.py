
import geopandas as gpd
import pandas as pd
import rasterio
from rasterstats import zonal_stats
from shapely import wkt
import matplotlib.pyplot as plt
import streamlit as st
import os
import numpy as np
import io
from matplotlib.colors import ListedColormap

st.set_page_config(page_title="Khartoum Flood Dashboard", layout="wide")

# --- Upload custom building data ---
uploaded_file = st.file_uploader("📤 Upload Your Building CSV", type=["csv"])
if uploaded_file:
    try:
        buildings_df = pd.read_csv(uploaded_file)
        st.success("✅ Custom building data loaded.")
    except Exception as e:
        st.error(f"❌ Failed to read uploaded file: {e}")
        st.stop()
else:
    try:
        buildings_df = pd.read_csv("data/buildings.csv", encoding='utf-8')
        st.subheader("📍 Sample Building Data")
        st.dataframe(buildings_df.head())
    except Exception as e:
        st.error(f"❌ Error loading buildings CSV: {e}")
        st.stop()

# --- Load Khartoum boundary ---
try:
    sudan_gdf = gpd.read_file("data/Khartoum.shp").to_crs("EPSG:4326")
    khartoum_gdf = sudan_gdf.copy()
except Exception as e:
    st.error(f"❌ Error loading Khartoum shapefile: {e}")
    st.stop()

# --- Parse and clean WKT geometry ---
if 'geometry' in buildings_df.columns:
    try:
        buildings_df['geometry'] = buildings_df['geometry'].astype(str).str.strip().str.replace('"', '')
        valid_wkt = buildings_df['geometry'].apply(lambda x: isinstance(x, str) and x.startswith(('POINT', 'POLYGON', 'MULTIPOLYGON')))
        buildings_df = buildings_df[valid_wkt].copy()
        buildings_df['geometry'] = buildings_df['geometry'].apply(wkt.loads)
        buildings_gdf = gpd.GeoDataFrame(buildings_df, geometry='geometry', crs='EPSG:4326')
    except Exception as e:
        st.error(f"❌ Error parsing WKT geometries: {e}")
        st.stop()
else:
    st.error("❌ 'geometry' column not found in buildings CSV.")
    st.stop()

# --- Clip buildings to Khartoum ---
try:
    buildings_in_khartoum = gpd.sjoin(buildings_gdf, khartoum_gdf, how='inner', predicate='intersects')
except Exception as e:
    st.error(f"❌ Error clipping buildings to Khartoum: {e}")
    st.stop()

# --- Efficient zonal stats function ---
def get_flooded_buildings_chunked(flood_path, buildings_gdf, chunk_size=50000):
    flooded_chunks = []
    buildings_gdf = buildings_gdf.to_crs("EPSG:4326")
    for i in range(0, len(buildings_gdf), chunk_size):
        chunk = buildings_gdf.iloc[i:i+chunk_size]
        try:
            stats = zonal_stats(chunk, flood_path, stats=["max"], nodata=0)
            flooded_idx = [i for i, s in enumerate(stats) if s and s.get("max") == 1]
            flooded_chunks.append(chunk.iloc[flooded_idx])
        except Exception as e:
            st.warning(f"⚠️ Zonal stats failed for chunk {i}: {e}")
    return pd.concat(flooded_chunks).to_crs("EPSG:4326") if flooded_chunks else gpd.GeoDataFrame(columns=buildings_gdf.columns)

# --- Load flood masks by date ---
flood_files = {
    "2020-08-30": "data/FloodMask_2020-08-30_MNDWI.tif",
    "2020-09-15": "data/FloodMask_2020-09-15_MNDWI.tif"
}

flooded_by_date = {}
for date_str, path in flood_files.items():
    if os.path.exists(path):
        try:
            flooded_by_date[date_str] = get_flooded_buildings_chunked(path, buildings_in_khartoum)
        except Exception as e:
            st.warning(f"⚠️ Error processing flood mask for {date_str}: {e}")
    else:
        st.warning(f"📁 Flood mask not found for {date_str}: {path}")

if not flooded_by_date:
    st.warning("⚠️ No flood data available.")
    st.stop()

# --- Raster plotting helper ---
def plot_flood_raster(ax, raster_path, scale_factor=0.1):
    try:
        with rasterio.open(raster_path) as src:
            st.write(f"✅ Raster opened: {raster_path}")
            new_height = int(src.height * scale_factor)
            new_width = int(src.width * scale_factor)
            flood_data = src.read(
                1,
                out_shape=(new_height, new_width),
                resampling=rasterio.enums.Resampling.nearest
            )
            extent = [src.bounds.left, src.bounds.right, src.bounds.bottom, src.bounds.top]
            cmap = ListedColormap(['none', 'blue'])  # dark blue for flood
            ax.imshow(flood_data, extent=extent, cmap=cmap, vmin=0, vmax=1, alpha=0.6)
    except Exception as e:
        st.warning(f"⚠️ Could not plot flood raster: {e}")

# --- Streamlit UI ---
st.title("🌊 Flood Impact on Buildings in Khartoum (Date-Specific)")
selected_date = st.selectbox("📅 Select Flood Date", sorted(flooded_by_date.keys()))
flooded = flooded_by_date[selected_date]

# 📊 Summary Metrics
with st.expander("📊 Summary Metrics", expanded=True):
    total_buildings = len(buildings_in_khartoum)
    flooded_count = len(flooded)
    percent_affected = round((flooded_count / total_buildings) * 100, 2)
    avg_area = buildings_in_khartoum.geometry.area.mean()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Buildings", total_buildings)
    col2.metric(f"Flooded on {selected_date}", flooded_count)
    col3.metric("Flooded %", f"{percent_affected}%")

# 📈 Multi-Date Comparison
with st.expander("📈 Flood Trend", expanded=False):
    trend_data = pd.DataFrame({
        "Date": list(flooded_by_date.keys()),
        "Flooded Buildings": [len(flooded_by_date[d]) for d in flooded_by_date]
    })
    st.bar_chart(trend_data.set_index("Date"))

# --- View toggle ---
view_mode = st.radio("🗺️ Select Building View Mode", ["Polygons", "Centroids"])
if view_mode == "Centroids":
    buildings_to_plot = buildings_in_khartoum.copy()
    buildings_to_plot["geometry"] = buildings_to_plot.centroid
    flooded_to_plot = flooded.copy()
    flooded_to_plot["geometry"] = flooded_to_plot.centroid
else:
    buildings_to_plot = buildings_in_khartoum
    flooded_to_plot = flooded

# --- Plotting ---
try:
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('none')

    if not khartoum_gdf.empty:
        khartoum_gdf.plot(ax=ax, edgecolor='black', facecolor='none', linewidth=0.5)

    scale_factor = st.slider("🧭 Raster Resolution", 0.05, 1.0, 0.1)

    if selected_date in flood_files and os.path.exists(flood_files[selected_date]):
        plot_flood_raster(ax, flood_files[selected_date], scale_factor)

    if not buildings_to_plot.empty:
        buildings_to_plot.plot(
            ax=ax,
            color='none',
            edgecolor='grey',
            linewidth=0.05,
            alpha=0.6,
            label='All Buildings'
        )

    if not flooded_to_plot.empty and flooded_to_plot.geometry.notnull().all():
        flooded_to_plot.plot(
            ax=ax,
            color='red',
            edgecolor='red',
            linewidth=0.05,
            label='Flooded Buildings'
        )

    ax.set_title(f"Flood Impact on {selected_date}", fontsize=16)
    ax.axis('off')
    ax.legend()
    st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Plotting failed: {e}")

# --- Optional download ---
st.download_button(
    label=f"Download Flooded Buildings ({selected_date})",
    data=flooded.to_csv(index=False),
    file_name=f"flooded_buildings_{selected_date}.csv",
    mime="text/csv"
)

buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight")
buf.seek(0)

st.download_button(
    label=f"📥 Download Plot ({selected_date}) as PNG",
    data=buf,
    file_name=f"flood_plot_{selected_date}.png",
    mime="image/png"
)

# --- Team Section ---
st.markdown("---")
st.subheader(" 🧑‍💼🧠 Meet the Team")

team_members = [
    {
        "name": "Mohamed Fadlelseed",
        "photo": "https://raw.githubusercontent.com/Moejezo95/khartoum-flood-dashboard/main/assets/me.jpg",
        "linkedin": "https://www.linkedin.com/in/mohamed-fadlelseed-98b015209/"
    },
     {
         "name": "Abreham Ashebir",
         "photo": "https://raw.githubusercontent.com/Moejezo95/khartoum-flood-dashboard/main/assets/abra.jpg",
         "linkedin": "https://www.linkedin.com/in/abreham-ashebir/"
     },
     {
         "name": "Mahmoud Abdi",
         "photo": "https://raw.githubusercontent.com/Moejezo95/khartoum-flood-dashboard/main/assets/abdi.jpg",
         "linkedin": "https://www.linkedin.com/in/mahamoud-abdi-abdillahi/"
     },
     {
         "name": "Muktar Abdinasir",
         "photo": "https://raw.githubusercontent.com/Moejezo95/khartoum-flood-dashboard/main/assets/mukh.jpg",
         "linkedin": "https://www.linkedin.com/in/muktar-abdinasir-salah-2689a3266/"
     }
]

cols = st.columns(len(team_members))
for col, member in zip(cols, team_members):
    col.image(member["photo"], width=160)
    col.markdown(f"**{member['name']}**")
    linkedin_icon = "https://cdn-icons-png.flaticon.com/512/174/174857.png"
    col.markdown(
    f'<a href="{member["linkedin"]}" target="_blank"><img src="{linkedin_icon}" width="24" style="margin-top:4px;"></a>',
    unsafe_allow_html=True
)
