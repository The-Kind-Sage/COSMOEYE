import os
import h5py
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Import custom processing engines
from preprocess import compute_spectral_ndvi
from spatial_math import extract_spatial_metrics
from generate_reports import execute_offline_report_generator

# Page configurations
st.set_page_config(
    page_title="COSMOS-EYE Control Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header interface
st.title("COSMOS-EYE: Regional Disaster Risk Reduction Dashboard")
st.caption("Change Observation and Satellite Monitoring Of Slopes — Localized Technical Control Room (Nepal)")
st.hr()

# Sidebar: File loading and selection
st.sidebar.header("Observation Vector Directory")
test_img_dir = "./datasets/TestData/img/"

if not os.path.exists(test_img_dir):
    st.sidebar.error("Data directory path missing. Check your extracted paths layout.")
    test_files = []
else:
    test_files = sorted([f for f in os.listdir(test_img_dir) if f.endswith(".h5")])

if len(test_files) == 0:
    st.sidebar.warning("No data matrices found. Add .h5 files to TestData/img/")
    selected_file = None
else:
    selected_file = st.sidebar.selectbox("Target Observation Tile Patch:", test_files)
    st.sidebar.success(f"Loaded target matrix item: {selected_file}")

# Main dashboard logic
if selected_file:
    target_absolute_file_path = os.path.join(test_img_dir, selected_file)

    # Calculate NDVI (Step 7/8)
    ndvi_matrix = compute_spectral_ndvi(target_absolute_file_path)
    
    # Extract spatial metrics (Step 14/15/16)
    simulated_prediction_layer = (ndvi_matrix < 0.15).astype(np.uint8)
    spatial_hazards_list = extract_spatial_metrics(simulated_prediction_layer, csv_path="local_highways.csv")

    # Build the prompt payload
    if spatial_hazards_list:
        primary_hazard = max(spatial_hazards_list, key=lambda x: x["surface_area_sqm"])
        prompt_string = (
            f"INPUT_VEC: District=Sindhupalchok; "
            f"Area={primary_hazard['surface_area_sqm']}sqm; "
            f"Nearest_Road={primary_hazard['nearest_infrastructure']}; "
            f"Blocked={primary_hazard['infrastructure_blocked']};"
        )
    else:
        prompt_string = "INPUT_VEC: District=Sindhupalchok; Anomaly=None; Status=Clear;"

    # Generate the NLP report (Step 20)
    natural_language_bulletin, _ = execute_offline_report_generator(prompt_string)

    # Layout columns
    left_column, right_column = st.columns([1, 1])

    # Left Column: NDVI Visualization
    with left_column:
        st.header("Spectral Feature Visualizations")
        
        fig, ax = plt.subplots(figsize=(6, 5))
        heatmap = ax.imshow(ndvi_matrix, cmap="RdYlGn", vmin=-1.0, vmax=1.0)
        ax.set_title("Manually Calculated NDVI Grid Matrix (Step 7)")
        ax.axis("off")
        fig.colorbar(heatmap, ax=ax, label="Vegetation Chlorophyll Signature Index Scale")
        
        st.pyplot(fig)
        st.caption("Interpretation Guide: Red/Yellow anomalies indicate exposed mud, debris flows, or landslide tracks.")

    # Right Column: Geospatial Statistics
    with right_column:
        st.header("Extracted Geospatial Metrics")
        
        if len(spatial_hazards_list) == 0:
            st.metric(label="System Status Verdict", value="Baseline Clear", delta="No Hazard Anomaly Detected")
            st.info("Geospatial calculations confirm terrain conditions fall within safe parameters.")
        else:
            primary = spatial_hazards_list[0]
            
            # KPI metric cards
            stat_col1, stat_col2 = st.columns(2)
            with stat_col1:
                st.metric(
                    label="Landslide Slope Surface Area", 
                    value=f"{primary['surface_area_sqm']:,} sqm",
                    delta=f"{primary['surface_area_sqm'] / 10000:.2f} Hectares",
                    delta_color="inverse"
                )
            with stat_col2:
                st.metric(
                    label="Critical Infrastructure Blockage Flag", 
                    value=f"Blocked: {primary['infrastructure_blocked']}",
                    delta=f"Nearest Transport: {primary['nearest_infrastructure']}",
                    delta_color="normal" if primary['infrastructure_blocked'] == "No" else "inverse"
                )

            # Raw anomaly log table
            st.subheader("Isolated Anomaly Database Records (Step 14 & 15)")
            st.dataframe(spatial_hazards_list, use_container_width=True)

    # Bottom Layout: Natural Language Report Text Output
    st.header("Decoded Last-Mile Intelligence Brief (Step 20 Output)")
    st.text_area(
        label="Automated Natural Language Bulletin Text Stream (NDRRMA Operational Format)",
        value=natural_language_bulletin,
        height=180,
        disabled=True
    )

else:
    st.info("Please verify data directories placement or select a file item in the sidebar control panel to wake up dashboard operations.")