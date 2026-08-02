import os
import sys
import h5py
import numpy as np
import streamlit as st

# Devanagari needs a UTF-8 console/output channel
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from predict import (execute_vision_inference_pass, resolve_image_path,
                     get_post_rgb, save_insight_panels,
                     PIXEL_AREA_SQM, INSIGHT_SCHEMA_VERSION)
from dataset import NDVI_CHANNEL, POST_STACK


@st.cache_data(show_spinner="Analyzing tile...", ttl=3600)
def run_insight(name, weights_mtime, img_mtime, schema_version):
    """Full inference, cached per tile. Keyed by the tile's own mtime, the
    weights mtime, and the insight schema version so results auto-invalidate
    when the tile, weights, or report code change. Insight panels are
    rendered to PNG files (reliable in every browser) and their paths are
    returned with the results."""
    spatial_metrics, binary_mask, insight = execute_vision_inference_pass(
        name, save_png=False, verbose=False)
    panel_paths = save_insight_panels(
        name, get_post_rgb(name), binary_mask, insight["prob_img"])
    return spatial_metrics, binary_mask, insight, panel_paths


def get_insight(name, weights_mtime, img_mtime, schema_version):
    """run_insight() with a guard against panel PNGs that no longer exist.

    The cache stores PATHS, not image bytes, so deleting prediction_overlays/
    (e.g. to force a re-render) leaves the cache handing back paths to missing
    files and st.image raises MediaFileStorageError. Re-render whenever any
    cached path has gone missing, then fall back to clearing the entry so the
    whole inference is redone.
    """
    result = run_insight(name, weights_mtime, img_mtime, schema_version)
    spatial_metrics, binary_mask, insight, panel_paths = result
    if panel_paths and all(os.path.exists(p) for p in panel_paths):
        return result

    # Cheap path: the inference result is still valid, only the PNGs vanished.
    try:
        panel_paths = save_insight_panels(
            name, get_post_rgb(name), binary_mask, insight["prob_img"])
        if panel_paths and all(os.path.exists(p) for p in panel_paths):
            return spatial_metrics, binary_mask, insight, panel_paths
    except Exception:
        pass

    # Last resort: drop the cached entry and recompute from scratch.
    run_insight.clear()
    return run_insight(name, weights_mtime, img_mtime, schema_version)

st.set_page_config(page_title="COSMOS-EYE Landslide Insight Dashboard",
                   page_icon="🛰️", layout="wide")
st.title("COSMOS-EYE: Landslide Insight Dashboard")
st.caption("Change Observation and Satellite Monitoring Of Slopes — Nepal")
st.divider()


@st.cache_data(ttl=300)
def list_real_tiles():
    """All real h5 tiles available in TestData/img (and TrainData if any)."""
    dirs = ["./datasets/TestData/img/", "./datasets/TrainData/img/"]
    tiles = []
    for d in dirs:
        if os.path.isdir(d):
            tiles.extend(
                f for f in sorted(os.listdir(d))
                if f.endswith(".h5") and not f.startswith("dummy")
            )
    return sorted(set(tiles))


def show_insight(insight):
    """Render the insight record + bilingual verdict in the sidebar."""
    ne = {"Landslide pixels": "पहिरो पिक्सेल",
          "Estimated area": "अनुमानित क्षेत्रफल",
          "sq m": "वर्ग मिटर", "ha": "हेक्टर",
          "largest body": "सबैभन्दा ठूलो पहिरो क्षेत्र",
          "confidence": "विश्वसनीयता",
          "Detected NDVI": "पत्ता लागेको NDVI",
          "Scene NDVI": "दृश्यको NDVI",
          "LANDSLIDE DETECTED": "पहिरो पत्ता लाग्यो",
          "NO landslide detected": "पहिरो पत्ता लागेन"}

    if insight["detected_pixels"] == 0:
        st.subheader(f"Verdict: {ne['NO landslide detected']}")
    else:
        st.subheader(f"Verdict: {ne['LANDSLIDE DETECTED']} — "
                     f"{insight['n_blobs']} region(s)")
        c1, c2, c3 = st.columns(3)
        c1.metric("Landslide pixels | पहिरो पिक्सेल",
                  f"{insight['detected_pixels']:,}",
                  f"{insight['scene_coverage_pct']:.2f}% of scene")
        c2.metric("Area | क्षेत्रफल",
                  f"{insight['area_sqm']:,.0f} m²",
                  f"{insight['area_ha']:.2f} ha")
        c3.metric("Largest body | ठूलो पहिरो",
                  f"{insight['largest_blob_area_sqm']:,.0f} m²")

        c4, c5 = st.columns(2)
        c4.metric("Detection confidence | विश्वसनीयता",
                  f"mean {insight['conf_mean']:.3f}",
                  f"max {insight['conf_max']:.3f}")
        c5.metric("Detected NDVI | पत्ता लागेको NDVI",
                  f"{insight['ndvi_mean_det']:.3f}",
                  "(bare ground = low)")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Target Image")
    tiles = list_real_tiles()
    selection = st.selectbox(
        "Select a real tile", tiles,
        help="Searchable — all TrainData/TestData tiles") if tiles else None

    uploaded = st.file_uploader("Or upload a custom .h5 file", type=["h5"])
    custom_path = None
    if uploaded is not None:
        # Stage uploads into TestData/img so the inference path resolution
        # (TestData then TrainData) can find them by name.
        os.makedirs("./datasets/TestData/img/", exist_ok=True)
        custom_path = os.path.join("./datasets/TestData/img/", uploaded.name)
        with open(custom_path, "wb") as fh:
            fh.write(uploaded.read())

    run_btn = st.button("Run landslide insight", type="primary")

target_name = uploaded.name if uploaded is not None else (selection or "dummy_test_patch.h5")

# ---------------------------------------------------------------- main panel
if run_btn:
    try:
        weights_mtime = os.path.getmtime("landslide_unet_weights.pth")
        img_mtime = os.path.getmtime(resolve_image_path(target_name))
        # spatial_metrics is still produced by the inference pass, but the
        # dashboard no longer renders it as a table.
        (_spatial_metrics, binary_mask, insight,
         panel_paths) = get_insight(target_name, weights_mtime, img_mtime,
                                    INSIGHT_SCHEMA_VERSION)
    except FileNotFoundError as e:
        st.error(f"Cannot load image: {e}")
        st.stop()

    st.success("Inference complete")
    show_insight(insight)

    # -------------------------------------------------- visual panels
    st.divider()
    st.subheader("Visual panels | दृश्य प्यानलहरू")
    panel_captions = ["Detection overlay (red = landslide)",
                      "Model probability (0-1)"]
    columns = st.columns(len(panel_captions))
    for column, index, caption in zip(columns, range(len(panel_captions)),
                                      panel_captions):
        with column:
            # Existence is re-checked here as well: st.image raises
            # MediaFileStorageError on a missing path, which aborts the whole
            # page rather than just the one panel.
            if index < len(panel_paths) and os.path.exists(panel_paths[index]):
                st.image(panel_paths[index], width="stretch", caption=caption)
            else:
                st.info("No panel rendered")

    # ---------------------------------------------- reports (separate)
    st.divider()
    from predict import build_insight_report
    en_report = build_insight_report(target_name, insight, lang="en")
    ne_report = build_insight_report(target_name, insight, lang="ne")
    tab_en, tab_ne = st.tabs(["English Report", "नेपाली प्रतिवेदन"])
    with tab_en:
        st.markdown("```\n" + en_report + "\n```")
    with tab_ne:
        st.markdown("```\n" + ne_report + "\n```")
else:
    st.info("Select a tile (or upload an .h5) in the sidebar and click "
            "**Run landslide insight**.")

if custom_path is not None and os.path.exists(custom_path):
    os.unlink(custom_path)
