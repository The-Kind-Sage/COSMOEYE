"""Download the pretrained COSMOS-EYE landslide model from GitHub Releases.

The model weights (.pth, ~31 MB) and the tuned threshold JSON are too large
for git, so they are shipped as GitHub Release assets instead. This script
fetches the latest release and installs the files into models/ so the
dashboard and inference work WITHOUT training anything.

Usage:
    python tools/download_weights.py
"""
import json
import os
import sys
import urllib.request

REPO = "The-Kind-Sage/COSMOEYE"
API_URL = f"https://api.github.com/repos/{REPO}/releases/latest"
REQUIRED_ASSETS = ["landslide_unet_weights.pth", "landslide_best_threshold.json"]
MODELS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "models"))


def main():
    print(f"Looking up the latest COSMOS-EYE release ({REPO})...")
    try:
        with urllib.request.urlopen(API_URL, timeout=30) as resp:
            release = json.load(resp)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            sys.exit(
                "ERROR: No release found on GitHub yet.\n\n"
                "To publish the pretrained model:\n"
                "  1. Open https://github.com/The-Kind-Sage/COSMOEYE/releases/new\n"
                "  2. Tag: v1.0.0  |  Title: Pretrained model weights\n"
                "  3. Attach these files:\n"
                "       models/landslide_unet_weights.pth\n"
                "       models/landslide_best_threshold.json\n"
                "  4. Click \"Publish release\", then run this script again."
            )
        sys.exit(f"ERROR: GitHub API request failed: {exc}")
    except urllib.error.URLError as exc:
        sys.exit(f"ERROR: no internet connection: {exc}")

    release_name = release.get("name") or release.get("tag_name")
    assets = {a["name"]: a["browser_download_url"] for a in release.get("assets", [])}

    os.makedirs(MODELS_DIR, exist_ok=True)
    for name in REQUIRED_ASSETS:
        url = assets.get(name)
        if url is None:
            print(f"  WARNING: {name} not attached to release {release_name} - skipped")
            continue
        dest = os.path.join(MODELS_DIR, name)
        print(f"  Downloading {name} ({release_name})...")
        urllib.request.urlretrieve(url, dest)
        size_mb = os.path.getsize(dest) / (1024 * 1024)
        if name.endswith(".pth") and size_mb < 1.0:
            os.remove(dest)
            sys.exit(f"ERROR: {name} is only {size_mb:.1f} MB - looks like an HTML "
                     f"error page instead of weights. Check the release URL.")
        print(f"    -> {dest} ({size_mb:.1f} MB)")

    print("\nDone. You can now run:")
    print("    streamlit run app.py")


if __name__ == "__main__":
    main()