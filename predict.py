import os
import csv
import json
import h5py
import numpy as np
import torch
import cv2
import functools

from model import CustomUNet
from dataset import (normalize_standardized_image, stack_temporal_stacks,
                     NORM_STATS_PATH, NDVI_CHANNEL, POST_STACK)
from spatial_math import extract_spatial_metrics, check_infrastructure_blockage

# Number of temporal windows the network expects (PRE / POST / LATE)
EXPECTED_WINDOWS = 3
# Must match train.py: the tuned threshold is only valid with the gate applied
NDVI_GATE_THRESHOLD = 0.48
# 10 m ground resolution -> every pixel covers 100 m^2
PIXEL_AREA_SQM = 100.0

# Bump whenever the insight dict gains/loses fields or report logic changes,
# so Streamlit caches keyed on this version are invalidated automatically.
INSIGHT_SCHEMA_VERSION = 19


def panel_title_style():
    """One shared title style for every rendered panel, so the PNGs in the
    Streamlit column layout read as a single set instead of separate
    matplotlib defaults.

    Deliberately a function, not a module-level dict: Streamlit's file
    watcher re-executes modules on save, and a partially-reloaded `predict`
    could expose the render functions while the constant they closed over was
    still missing, raising NameError deep inside panel rendering. Resolving
    the style at call time removes that window entirely.
    """
    return {
        "fontsize": 15,
        "fontweight": "bold",
        "fontfamily": "DejaVu Sans",
        "color": "#1a1a1a",
        "pad": 12,
    }


# Bilingual report labels (English | Nepali Devanagari)
NEP = {
    "LANDSLIDE INSIGHT": "पहिरो विश्लेषण",
    "Verdict": "निष्कर्ष",
    "LANDSLIDE DETECTED": "पहिरो पत्ता लाग्यो",
    "NO landslide detected": "पहिरो पत्ता लागेन",
    "separate region(s)": "छुट्टा-छुट्टै क्षेत्रहरू",
    "Landslide pixels": "पहिरो पिक्सेल",
    "% of scene": "दृश्यको प्रतिशत",
    "Estimated area": "अनुमानित क्षेत्रफल",
    "sq m": "वर्ग मिटर",
    "ha": "हेक्टर",
    "Largest body": "सबैभन्दा ठूलो पहिरो क्षेत्र",
    "Detection confidence": "पत्ता लागेको विश्वसनीयता",
    "mean": "औसत",
    "max": "अधिकतम",
    "Detected NDVI": "पत्ता लागेको NDVI",
    "bare ground = low NDVI": "खुला जमिन = कम NDVI",
    "Scene NDVI": "दृश्यको NDVI",
    "min": "न्यूनतम",
    "Per-region landslide probability": "क्षेत्रअनुसार पहिरोको सम्भाव्यता",
    "Region": "क्षेत्र",
    "probability": "सम्भाव्यता",
    "peak": "उच्चतम",
    "Gate-excluded pixels": "NDVI गेटले बहिष्कृत पिक्सेल",
    "max prob": "अधिकतम प्रायिकता",
    "Peak model probability": "मोडेलको अधिकतम प्रायिकता",
    "detection threshold": "पत्ता लगाउने थ्रेसहोल्ड",
    "below detection threshold": "पत्ता लगाउने थ्रेसहोल्डभन्दा कम",
    "removed by NDVI gate": "NDVI गेटले हटाइयो",
    # Road blockage
    "Nearest road": "नजिकको सडक",
    "Road blocked": "सडक अवरुद्ध",
    "Road clear": "सडक खुला",
    "within": "भित्र",
    "m of landslide": "मिटर पहिरो नजिक",
    # Cloud / occlusion warning
    "WARNING: Low valid-observation coverage": "चेतावनी: कम वैध अवलोकन कवरेज",
    "valid observations in POST window": "POST विन्डोमा वैध अवलोकन",
    "Results may be unreliable due to cloud/occlusion": "बादल/ओकुलेसनका कारण परिणाम अविश्वसनीय हुन सक्छ",
    # Human-in-the-loop triage disclaimer
    "TRIAGE AID ONLY": "केवल ट्रायज सहायता",
    "triage_disclaimer_en": (
        "IMPORTANT: This output is a machine-learning triage aid intended to "
        "help prioritise field surveys — it is NOT a confirmed hazard assessment. "
        "All detections must be verified by a qualified geohazard specialist "
        "before any emergency response or infrastructure decisions are made."
    ),
    "triage_disclaimer_ne": (
        "महत्त्वपूर्ण: यो आउटपुट एक मेसिन-लर्निङ ट्रायज सहायता हो जसले "
        "क्षेत्र सर्वेक्षणलाई प्राथमिकता दिन मद्दत गर्छ — यो पुष्टि भएको "
        "खतरा मूल्याङ्कन होइन। कुनै पनि आपतकालीन प्रतिक्रिया वा पूर्वाधार "
        "निर्णय गर्नु अघि सबै पत्ता लगाइएका क्षेत्रहरू योग्य भूगर्भ विशेषज्ञद्वारा "
        "प्रमाणित गरिनुपर्छ।"
    ),
}


def build_standardized_input(raw_matrix):
    """Convert any supported input layout into the standardized temporal-stack layout.

    - 14 channels (legacy single-frame SEN12 tiles): used as-is.
    - 4D (N, H, W, 14) (temporal-stack tiles): PRE/POST/LATE windows, as-is.
    Single-frame inputs become a one-window container so the temporal
    network can still score them.
    """
    if raw_matrix.ndim == 4:
        return _fit_window_count(raw_matrix.astype(np.float32))

    height, width, channels = raw_matrix.shape

    if channels == 14:
        fused_matrix = raw_matrix
    else:
        raise ValueError(f"Unsupported channel count: {channels} (expected 14 or a temporal stack)")

    return _fit_window_count(fused_matrix[None])


def _fit_window_count(stacks):
    """Force the container to the EXPECTED_WINDOWS the network was built for."""
    n_windows = stacks.shape[0]
    if n_windows == EXPECTED_WINDOWS:
        return stacks
    if n_windows == 1:
        return np.repeat(stacks, EXPECTED_WINDOWS, axis=0)
    if n_windows > EXPECTED_WINDOWS:
        return stacks[:EXPECTED_WINDOWS]
    pad = np.repeat(stacks[-1:], EXPECTED_WINDOWS - n_windows, axis=0)
    return np.concatenate([stacks, pad], axis=0)


@functools.lru_cache(maxsize=1)
def load_normalization_stats():
    """Load the dataset-global z-score stats used during training."""
    if os.path.exists(NORM_STATS_PATH):
        with np.load(NORM_STATS_PATH) as d:
            return d["mean"].astype(np.float32), d["std"].astype(np.float32)
    return None, None


def infer_base_filters(state_dict, default=64):
    """Read the model width straight out of the checkpoint.

    The default width changed from 96 to 64 (the wider build overfit this
    dataset). Hard-coding either value makes checkpoints from the other build
    fail to load with an opaque shape error, so the width is read from the
    first encoder conv, whose weight is (base_filters, in_channels, 3, 3).
    """
    weight = state_dict.get("down1.conv.0.weight")
    return int(weight.shape[0]) if weight is not None else default


@functools.lru_cache(maxsize=2)
def _build_model_for(weights_mtime):
    """Build the U-Net and load weights. Keyed by weights file mtime so the
    model is built once per weights version and never re-read on every call
    (Streamlit reruns / repeated inferences)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load("landslide_unet_weights.pth", map_location=device,
                       weights_only=True)
    base_filters = infer_base_filters(state)
    unet = CustomUNet(in_channels=42, out_channels=1, base_filters=base_filters)
    unet.load_state_dict(state)
    return unet.to(device).eval()


def get_cached_model():
    """Return the cached inference model (caches across reruns)."""
    return _build_model_for(os.path.getmtime("landslide_unet_weights.pth"))


def tta_probabilities(model, image_tensor, device):
    """Average sigmoid probabilities over identity + 3 flips."""
    views = [
        (image_tensor, ()),
        (torch.flip(image_tensor, dims=[3]), (3,)),
        (torch.flip(image_tensor, dims=[2]), (2,)),
        (torch.flip(image_tensor, dims=[2, 3]), (2, 3)),
    ]
    probs = None
    with torch.no_grad():
        for view, flip_dims in views:
            logits = model(view)
            logits = torch.flip(logits, dims=list(flip_dims)) if flip_dims else logits
            prob = torch.sigmoid(logits.float())
            probs = prob if probs is None else probs + prob
    return probs / len(views)


def analyze_detections(binary_mask, probability_mask, raw_post_ndvi):
    """Extract the per-image insight record from a prediction.

    Returns a dict with landslide pixel/area stats, the largest connected
    blob, confidence profile of detections, and NDVI of the detected region.
    """
    total_px = binary_mask.size
    detected_px = int(binary_mask.sum())
    area_sqm = detected_px * PIXEL_AREA_SQM

    # Largest connected component (the dominant landslide body)
    n_blobs, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, 8)
    largest_area = 0
    if n_blobs > 1:  # index 0 is the background
        largest = stats[1:, cv2.CC_STAT_AREA].argmax()
        largest_area = int(stats[1:][largest][cv2.CC_STAT_AREA]) * PIXEL_AREA_SQM

    # Per-region confidence. The scene-wide conf_mean/conf_max below average
    # every detected pixel together, which says nothing about how confident
    # any INDIVIDUAL landslide body is - a large certain slide and a marginal
    # speck collapse into one number. Reported per region, largest first.
    regions = []
    for blob_idx in range(1, n_blobs):
        blob = labels == blob_idx
        blob_probs = probability_mask[blob]
        regions.append({
            "area_sqm": int(stats[blob_idx, cv2.CC_STAT_AREA]) * PIXEL_AREA_SQM,
            "prob_mean_pct": float(blob_probs.mean()) * 100.0,
            "prob_max_pct": float(blob_probs.max()) * 100.0,
        })
    regions.sort(key=lambda r: r["area_sqm"], reverse=True)

    det_conf = probability_mask[binary_mask > 0]
    scene_ndvi = raw_post_ndvi
    det_ndvi = scene_ndvi[binary_mask > 0]

    return {
        "detected_pixels": detected_px,
        "scene_pixels": total_px,
        "scene_coverage_pct": 100.0 * detected_px / max(1, total_px),
        "area_sqm": area_sqm,
        "area_ha": area_sqm / 10000.0,
        "n_blobs": max(0, n_blobs - 1),
        "largest_blob_area_sqm": largest_area,
        "regions": regions,
        "conf_mean": float(det_conf.mean()) if det_conf.size else 0.0,
        "conf_max": float(det_conf.max()) if det_conf.size else 0.0,
        "ndvi_mean_det": float(det_ndvi.mean()) if det_ndvi.size else np.nan,
        "ndvi_min_scene": float(np.nanmin(scene_ndvi)),
        "ndvi_mean_scene": float(np.nanmean(scene_ndvi)),
        "prob_img": probability_mask.astype(np.float32),
    }



DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def to_devanagari_digits(text):
    """Rewrite ASCII digits as Devanagari numerals (0-9 -> ०-९).

    Separators are converted too: Nepali uses the same '.' decimal point and
    ',' thousands separator, so only the digit glyphs change.
    """
    return text.translate(DEVANAGARI_DIGITS)


def build_insight_report(sample_file_name, insight, lang="en"):
    """Return a complete per-image insight report as a string, entirely in
    one language: "en" (English) or "ne" (Nepali Devanagari).

    The Nepali report uses Devanagari numerals throughout, so it reads as
    Nepali rather than as Nepali labels with English numbers. The tile
    filename is the one exception - it is an identifier that has to stay
    typeable and matchable against files on disk."""
    tr = NEP if lang == "ne" else {k: k for k in NEP}
    lines = []

    # --- Triage disclaimer (top of every report) ---
    disclaimer_key = "triage_disclaimer_ne" if lang == "ne" else "triage_disclaimer_en"
    lines.append(f"[{tr['TRIAGE AID ONLY']}] {NEP[disclaimer_key]}")
    lines.append("")

    lines.append(f"{tr['LANDSLIDE INSIGHT']} — {sample_file_name}")

    # --- Cloud / occlusion warning ---
    valid_frac = insight.get("valid_obs_fraction", None)
    if valid_frac is not None and valid_frac < 0.80:
        warn_pct = f"{valid_frac * 100:.1f}%"
        if lang == "ne":
            warn_pct = to_devanagari_digits(warn_pct)
        lines.append(f"  *** {tr['WARNING: Low valid-observation coverage']}: "
                     f"{warn_pct} {tr['valid observations in POST window']} — "
                     f"{tr['Results may be unreliable due to cloud/occlusion']} ***")

    if insight["detected_pixels"] == 0:
        lines.append(f"{tr['Verdict']}: {tr['NO landslide detected']}")
        lines.append(f"  {tr['Scene NDVI']}: {tr['mean']} "
                     f"{insight['ndvi_mean_scene']:.3f} "
                     f"({tr['min']} {insight['ndvi_min_scene']:.3f})")
        peak = insight.get("peak_prob", 0.0)
        thr = insight.get("threshold", 0.5)
        note = (tr["below detection threshold"] if peak < thr
                else tr["removed by NDVI gate"])
        lines.append(f"  {tr['Peak model probability']}: {peak:.3f} "
                     f"({tr['detection threshold']} {thr:.2f} — {note})")
    else:
        lines.append(f"{tr['Verdict']}: {tr['LANDSLIDE DETECTED']} "
                     f"({insight['n_blobs']} {tr['separate region(s)']})")
        lines.append(f"  {tr['Landslide pixels']}: "
                     f"{insight['detected_pixels']:,} "
                     f"({insight['scene_coverage_pct']:.2f} {tr['% of scene']})")
        lines.append(f"  {tr['Estimated area']}: {insight['area_sqm']:,.0f} "
                     f"{tr['sq m']} ({insight['area_ha']:.2f} {tr['ha']})")
        lines.append(f"  {tr['Largest body']}: "
                     f"{insight['largest_blob_area_sqm']:,.0f} {tr['sq m']}")
        lines.append(f"  {tr['Detection confidence']}: {tr['mean']} "
                     f"{insight['conf_mean']:.3f} | {tr['max']} "
                     f"{insight['conf_max']:.3f}")
        lines.append(f"  {tr['Detected NDVI']}: {tr['mean']} "
                     f"{insight['ndvi_mean_det']:.3f} "
                     f"({tr['bare ground = low NDVI']})")
        regions = insight.get("regions", [])
        if regions:
            lines.append(f"  {tr['Per-region landslide probability']}:")
            for n, region in enumerate(regions, start=1):
                lines.append(f"    {tr['Region']} {n}: "
                             f"{region['prob_mean_pct']:.1f}% "
                             f"{tr['probability']} "
                             f"({tr['peak']} {region['prob_max_pct']:.1f}%) — "
                             f"{region['area_sqm']:,.0f} {tr['sq m']}")

        # --- Road proximity / blockage ---
        nearest_road = insight.get("nearest_road")
        road_blocked = insight.get("road_blocked", False)
        if nearest_road is not None:
            dist_m = insight.get("nearest_road_dist_m", None)
            if road_blocked and dist_m is not None:
                lines.append(f"  {tr['Road blocked']}: {nearest_road} "
                             f"({tr['within']} {dist_m:.0f} {tr['m of landslide']})")
            else:
                lines.append(f"  {tr['Nearest road']}: {nearest_road} "
                             f"— {tr['Road clear']}")

    if insight.get("gate_excluded_pixels", 0) > 0:
        lines.append(f"  {tr['Gate-excluded pixels']}: "
                     f"{insight['gate_excluded_pixels']:,} "
                     f"({tr['max prob']} {insight['gate_excluded_max_prob']:.3f})")
    if lang == "ne":
        # Header line (disclaimer) and tile filename line stay in ASCII so they
        # remain searchable and matchable against files on disk; only the
        # measurement lines get Devanagari digits.
        lines = [lines[0], lines[1]] + [to_devanagari_digits(l) for l in lines[2:]]
    report = "\n".join(lines)
    return report


def print_insight_report(sample_file_name, insight):
    """Print the English report, then the Nepali report, as two separate
    clearly-labeled sections."""
    en = build_insight_report(sample_file_name, insight, lang="en")
    ne = build_insight_report(sample_file_name, insight, lang="ne")
    print("\n==================== ENGLISH REPORT ====================")
    print(en)
    print("==================== नेपाली प्रतिवेदन ====================")
    print(ne)
    print("==========================================================\n")


def resolve_image_path(sample_file_name):
    """Locate a real image in TestData or TrainData."""
    candidates = [
        f"./datasets/TestData/img/{sample_file_name}",
        f"./datasets/TrainData/img/{sample_file_name}",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Image {sample_file_name} not found in TestData/img or TrainData/img"
    )


def resolve_mask_path(sample_file_name):
    """Locate a matching ground-truth mask, if one exists."""
    for path in (f"./datasets/TestData/mask/{sample_file_name}",
                 f"./datasets/TrainData/mask/{sample_file_name}"):
        if os.path.exists(path):
            return path
    return None


def save_overlay(sample_file_name, raw_post_rgb, binary_mask, probability_mask,
                 out_dir="prediction_overlays"):
    """Save a side-by-side overlay PNG (RGB, probability, binary mask)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return  # visualization is optional; insight data still works

    os.makedirs(out_dir, exist_ok=True)
    # Raw reflectance DN (0-10000) is outside imshow's expected range; scale
    # to [0,1] using a 2nd-98th percentile stretch so the RGB panels render
    # properly instead of clipping everything to white.
    rgb_norm = _percentile_stretch_rgb(raw_post_rgb)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_norm)
    axes[0].set_title("Natural-color satellite (POST window)", **panel_title_style())
    im = axes[1].imshow(probability_mask, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Model probability (0-1)", **panel_title_style())
    axes[2].imshow(rgb_norm)
    axes[2].imshow(mask_overlay_rgba(binary_mask))
    draw_mask_outline(axes[2], binary_mask)
    axes[2].set_title("Detection overlay (red = landslide)", **panel_title_style())
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    base = os.path.splitext(sample_file_name)[0]
    fig.savefig(os.path.join(out_dir, f"{base}_overlay.png"), dpi=120,
                bbox_inches="tight")
    plt.close(fig)


def mask_overlay_rgba(binary_mask, color=(1.0, 0.0, 0.0), alpha=1.0):
    """Build an explicit RGBA layer: solid `color` where the mask is set,
    fully transparent everywhere else.

    This replaces `imshow(masked_array, cmap="Reds")`, which rendered PINK
    rather than red. A binary mask masked to its positives holds a single
    value (1), so vmin == vmax, matplotlib's Normalize maps that value to
    0.0, and Reds(0.0) is #FFF5F0 - near-white. Blended at 55% over the
    satellite image it washed the whole detection area pale pink. Writing the
    RGBA values directly removes the colour-scale step entirely, so the
    colour cannot depend on how many distinct values the mask happens to have.
    """
    rgba = np.zeros(binary_mask.shape[:2] + (4,), dtype=np.float32)
    hit = binary_mask > 0
    rgba[hit, 0], rgba[hit, 1], rgba[hit, 2] = color
    rgba[hit, 3] = alpha
    return rgba


def draw_mask_outline(ax, binary_mask, color="#ff1a1a", linewidth=1.2):
    """Outline the mask boundary. Landslide blobs are tiny (median ~4x4 px),
    so a translucent fill alone is easy to miss at display size."""
    if binary_mask.any() and not binary_mask.all():
        ax.contour(binary_mask.astype(float), levels=[0.5],
                   colors=color, linewidths=linewidth)


def get_post_rgb(sample_file_name):
    """Return the POST-window RGB (B04/B03/B02) for a tile as (H, W, 3).
    Channel layout: 0=B02, 1=B03, 2=B04, ..., 13=NDVI."""
    with h5py.File(resolve_image_path(sample_file_name), "r") as f:
        raw_matrix = np.array(f["img"]).astype(np.float32)
    fused_matrix = build_standardized_input(raw_matrix)
    post_idx = min(POST_STACK, fused_matrix.shape[0] - 1)
    return fused_matrix[post_idx, :, :, [2, 1, 0]].transpose(1, 2, 0)


def _percentile_stretch_rgb(rgb):
    """Render raw reflectance DN as a Google-Maps-style natural-color image.

    The previous version stretched each band independently and then boosted
    saturation 50%, which pushed bare rock and soil (genuinely brown-grey,
    with red > blue > green reflectance) all the way into MAGENTA. Natural
    colour is restored here in four steps:

      1. Joint 2nd-98th percentile stretch - one contrast range for all
         three bands, so relative band brightness (i.e. hue) survives.
      2. Gray-world white balance - equalise the channel means, removing
         the scene-wide colour cast that produced the pink wash.
      3. Gamma + a mild saturation lift (vegetation reads green without
         over-driving soil into false colour).
      4. Magenta guard - magenta/pink is by definition green < min(red,
         blue). Lifting green to that floor collapses the magenta sector
         onto neutral grey and cannot touch greens, reds or browns.
    """
    import cv2
    rgb = rgb.astype(np.float32)
    # 1. joint stretch (scalar lo/hi, not per-channel) preserves hue
    # 1a. Dark-object subtraction: Rayleigh scattering adds a large additive
    #     offset to blue, less to green, least to red. Left in, it washes the
    #     whole scene toward pale lavender - the "hazy" look that consumer
    #     basemaps de-haze away. The darkest 1% of each band is assumed to be
    #     that path radiance and removed. Only 60% of it: full subtraction
    #     crushed blue far below red and tipped bare ground into salmon.
    haze = np.percentile(rgb, 1.0, axis=(0, 1), keepdims=True)
    rgb = np.maximum(rgb - 0.6 * haze, 0.0)

    # A 98th-percentile ceiling clipped ~2% of every scene to pure white;
    # 99.5 keeps bright bare ground as detail instead of a blown-out patch.
    lo = float(np.percentile(rgb, 2.0))
    hi = float(np.percentile(rgb, 99.5))
    stretched = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

    # 2. Full gray-world white balance. This is the load-bearing step for
    #    "no false colour": equal channel means means the scene as a whole is
    #    neutral, so no global pink or salmon cast can survive it. Applying it
    #    at partial strength let de-hazed blue stay low and bare ground read
    #    salmon, so it stays at full strength.
    means = stretched.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    stretched = np.clip(stretched * (gray / np.maximum(means, 1e-6)), 0.0, 1.0)

    # 3. gamma. 1/1.25 rather than 1/1.6: the stronger lift washed forest
    #    canopy out to pale grey-green instead of the deep green of a
    #    consumer satellite basemap.
    bright = np.power(stretched, 1.0 / 1.3)

    # 4. Contrast via an S-curve pivoted on the scene's own median, NOT CLAHE.
    #    CLAHE (even at clipLimit 1.3) equalises each 8x8 tile of what is a
    #    dark scene, which lifted median luminance 0.29 -> 0.45 and drove 6.5%
    #    of the frame to pure white. A global S-curve adds the same sense of
    #    contrast while leaving overall exposure where the stretch put it.
    pivot = float(np.median(bright))
    contrast = 1.12
    out = np.clip(pivot + (bright - pivot) * contrast, 0.0, 1.0)
    # soft highlight shoulder so bright bare ground keeps texture
    out = np.where(out > 0.85, 0.85 + (out - 0.85) * 0.55, out)

    # 5. saturation lift for vivid vegetation
    hsv = cv2.cvtColor((out * 255).astype(np.uint8),
                       cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.08, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8),
                       cv2.COLOR_HSV2RGB).astype(np.float32) / 255.0

    # 6. magenta guard - no pink is allowed to reach the panels
    out[..., 1] = np.maximum(out[..., 1], np.minimum(out[..., 0], out[..., 2]))
    return np.clip(out, 0.0, 1.0)


def _enhance_for_display(rgb_norm, scale=4):
    """Upscale the 128x128 tile for display and re-sharpen it.

    At native size a 10 m/px tile is a blocky 128 px thumbnail; matplotlib
    then does its own nearest/linear resampling into the figure, which is
    what made the panel look soft and pixelated next to a real basemap.
    Lanczos upscaling followed by an unsharp mask recovers crisp road and
    field edges at the size the panel is actually displayed.
    """
    import cv2
    h, w = rgb_norm.shape[:2]
    big = cv2.resize(rgb_norm, (w * scale, h * scale),
                     interpolation=cv2.INTER_LANCZOS4)
    # A tight radius sharpens the fine detail the Lanczos interpolation
    # smeared, which is what actually reads as "in focus". A wide radius
    # (sigma 1.8) only re-shaped large blobs and left the softness; a heavy
    # amount (1.75) tipped over into crunchy edge halos. sigma 0.8 at 1.45x
    # is the point where detail is crisp but the image still looks
    # photographic rather than over-processed.
    blur = cv2.GaussianBlur(big, (0, 0), sigmaX=0.8)
    sharp = cv2.addWeighted(big, 1.45, blur, -0.45, 0.0)
    return np.clip(sharp, 0.0, 1.0)


def save_insight_panels(sample_file_name, raw_post_rgb, binary_mask,
                        probability_mask, out_dir="prediction_overlays"):
    """Render the insight panels as separate PNG files and return their
    paths. Panel 1: RGB with the red detection overlay; Panel 2: probability
    map (jet). Files are always RGB-mode PNGs, so any browser/display path
    renders them reliably.

    The plain natural-color satellite panel was dropped: the detection
    overlay already shows the same imagery underneath the mask, so it was
    pure duplication in the dashboard's column layout."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(sample_file_name))[0]

    # Display-resolution basemap: upscale + re-sharpen so the panel reads like
    # a real satellite basemap instead of a 128 px thumbnail. The mask is
    # upscaled by the SAME factor with nearest-neighbour, so detection edges
    # stay pixel-exact and never bleed into neighbouring ground.
    scale = 4
    rgb_norm = _enhance_for_display(_percentile_stretch_rgb(raw_post_rgb), scale)
    mask_big = cv2.resize(binary_mask.astype(np.uint8),
                          (binary_mask.shape[1] * scale,
                           binary_mask.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)

    paths = []
    # Panel 1: natural-color RGB + red detection overlay
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.imshow(rgb_norm, interpolation="lanczos")
    ax.imshow(mask_overlay_rgba(mask_big), interpolation="nearest")
    draw_mask_outline(ax, mask_big, linewidth=1.8)
    ax.set_title("Detection overlay (red = landslide)", **panel_title_style())
    ax.axis("off")
    p1 = os.path.join(out_dir, f"{base}_overlay.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(p1)

    # Panel 2: probability map (same figure size, no tight-crop, so all
    # PNGs are exactly the same pixel dimensions and render identically)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    im = ax.imshow(probability_mask, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Model probability (0-1)", **panel_title_style())
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046)
    cbar.ax.tick_params(labelsize=11)
    p2 = os.path.join(out_dir, f"{base}_prob.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(p2)
    return paths


def execute_vision_inference_pass(sample_file_name, save_png=True, verbose=True,
                                   road_proximity_m=500.0):
    """Run the full inference pipeline on one tile.

    Parameters
    ----------
    sample_file_name : str
        Tile filename (resolved via TestData/TrainData search).
    save_png : bool
        Whether to write a side-by-side overlay PNG to prediction_overlays/.
    verbose : bool
        Print the bilingual insight report to stdout.
    road_proximity_m : float
        Distance threshold (metres) within which a detected landslide is
        considered to block a road.  Default 500 m (~5 pixels at 10 m/px).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_img_path = resolve_image_path(sample_file_name)
    vision_weights = "landslide_unet_weights.pth"

    if not os.path.exists(vision_weights):
        raise FileNotFoundError("Missing local brain weights file! Run train.py first.")

    with h5py.File(test_img_path, "r") as f:
        raw_matrix = np.array(f["img"]).astype(np.float32)

    fused_matrix = build_standardized_input(raw_matrix)

    # CRITICAL: apply the exact same normalization used during training,
    # otherwise the network receives out-of-distribution values.
    norm_mean, norm_std = load_normalization_stats()
    normalized = normalize_standardized_image(fused_matrix, norm_mean, norm_std)

    image_tensor = torch.from_numpy(
        stack_temporal_stacks(normalized)
    ).unsqueeze(0).to(device)

    unet_model = get_cached_model()
    unet_model.eval()

    probability_mask = tta_probabilities(unet_model, image_tensor, device).squeeze().cpu().numpy()

    # Use the IoU-tuned threshold saved during training (falls back to 0.5).
    # The ndvi_gate default is now False: the primary training metric is
    # raw/ungated, so the saved threshold belongs to the ungated distribution.
    threshold = 0.5
    gate_enabled = False
    gate_threshold = NDVI_GATE_THRESHOLD
    threshold_path = "landslide_best_threshold.json"
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as jf:
            saved = json.load(jf)
        threshold = saved.get("threshold", 0.5)
        gate_enabled = saved.get("ndvi_gate", False)
        gate_threshold = saved.get("ndvi_gate_threshold", NDVI_GATE_THRESHOLD)

    post_idx = min(POST_STACK, fused_matrix.shape[0] - 1)
    raw_post_ndvi = fused_matrix[post_idx, :, :, NDVI_CHANNEL]
    raw_probability_mask = probability_mask

    # --- Cloud / occlusion: valid-observation fraction for the POST window ---
    # Sentinel-2 scene classification marks cloud/shadow/snow with NDVI < -0.1
    # (deep negative reflectance ratios) or saturated bands.  A proxy that
    # works on whatever we have: pixels where ALL bands in the POST stack are
    # exactly 0 are fill / no-data (cloud-masked out by the pre-processor).
    post_stack_flat = fused_matrix[post_idx]          # (H, W, 14)
    invalid_px = np.all(post_stack_flat == 0.0, axis=-1)   # (H, W) bool
    valid_obs_fraction = float(1.0 - invalid_px.mean())

    gate_excluded_px = 0
    gate_excluded_max = 0.0
    if gate_enabled:
        would_detect = raw_probability_mask > threshold
        excluded = would_detect & (raw_post_ndvi >= gate_threshold)
        gate_excluded_px = int(excluded.sum())
        if gate_excluded_px:
            gate_excluded_max = float(raw_probability_mask[excluded].max())
        probability_mask = raw_probability_mask * (raw_post_ndvi < gate_threshold)

    binary_mask = (probability_mask > threshold).astype(np.uint8)

    insight = analyze_detections(binary_mask, probability_mask, raw_post_ndvi)
    insight["threshold"] = float(threshold)
    insight["gate_enabled"] = gate_enabled
    insight["gate_excluded_pixels"] = gate_excluded_px
    insight["gate_excluded_max_prob"] = gate_excluded_max
    insight["peak_prob"] = float(probability_mask.max())
    insight["valid_obs_fraction"] = valid_obs_fraction

    # --- Road proximity / blockage check ---
    blockage = check_infrastructure_blockage(binary_mask,
                                             proximity_m=road_proximity_m)
    insight["nearest_road"] = blockage.get("nearest_road")
    insight["nearest_road_dist_m"] = blockage.get("nearest_road_dist_m")
    insight["road_blocked"] = blockage.get("road_blocked", False)

    spatial_metrics = extract_spatial_metrics(binary_mask)

    if verbose:
        print("================== GEOSPATIAL ANALYSIS PASS COMPLETE ==================")
        print(f"Processed Target File: {sample_file_name}")
        print(f"Valid POST observations: {valid_obs_fraction * 100:.1f}%")
        if valid_obs_fraction < 0.80:
            print(f"  WARNING: low valid-obs coverage — cloud/occlusion may affect results")
        if blockage.get("road_blocked"):
            print(f"  ROAD BLOCKAGE: {blockage['nearest_road']} within "
                  f"{blockage['nearest_road_dist_m']:.0f} m")
        print(f"Discovered Hazard Anomalies Records: {spatial_metrics}")
        print_insight_report(sample_file_name, insight)
        print("=======================================================================\n")

    if save_png:
        post_rgb = fused_matrix[post_idx, :, :, [2, 1, 0]].transpose(1, 2, 0)
        save_overlay(sample_file_name, post_rgb, binary_mask, probability_mask)

    return spatial_metrics, binary_mask, insight


def batch_predict(test_img_dir="./datasets/TestData/img/",
                  out_csv="per_image_predictions.csv"):
    """Run the insight pipeline on EVERY test image and write a CSV table."""
    os.makedirs(test_img_dir, exist_ok=True)
    files = sorted(f for f in os.listdir(test_img_dir) if f.endswith(".h5"))
    if not files:
        print("No .h5 files in TestData/img/")
        return

    print(f"Batch inference over {len(files)} test images...")
    rows = []
    for i, fname in enumerate(files):
        try:
            _, _, ins = execute_vision_inference_pass(fname, save_png=True,
                                                      verbose=False)
            rows.append([fname, ins["detected_pixels"], ins["area_sqm"],
                         f"{ins['area_ha']:.2f}", ins["n_blobs"],
                         f"{ins['largest_blob_area_sqm']:,.0f}",
                         f"{ins['scene_coverage_pct']:.3f}",
                         f"{ins['conf_mean']:.3f}", f"{ins['conf_max']:.3f}",
                         f"{ins['ndvi_mean_det']:.3f}",
                         f"{ins['ndvi_mean_scene']:.3f}"])
        except Exception as exc:
            print(f"  FAILED {fname}: {exc}")
    if rows:
        with open(out_csv, "w", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["image", "detected_pixels", "area_sqm",
                             "area_ha", "n_regions", "largest_blob_sqm",
                             "scene_coverage_pct", "conf_mean", "conf_max",
                             "ndvi_mean_det", "ndvi_mean_scene"])
            writer.writerows(rows)
        print(f"Per-image results written to {out_csv} ({len(rows)} images)")


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        batch_predict()
    elif len(sys.argv) > 1:
        execute_vision_inference_pass(sys.argv[1])
    else:
        os.makedirs("./datasets/TestData/img/", exist_ok=True)
        mock_test_file = "dummy_test_patch.h5"
        mock_path = f"./datasets/TestData/img/{mock_test_file}"
        if not os.path.exists(mock_path):
            with h5py.File(mock_path, "w") as f:
                f.create_dataset("img", data=np.random.rand(128, 128, 14))
        try:
            execute_vision_inference_pass(mock_test_file)
        except FileNotFoundError as e:
            print(f"Skipped: {e}")
