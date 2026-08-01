"""Geo-lookup utility: convert Sen12Landslides .nc tile coordinates to
WGS84 latitude/longitude and print Google Maps links.

The raw tiles store their position in UTM meters (attributes are misnamed
center_lat/center_lon) plus a per-file `crs` attribute, e.g.:
    EPSG:32736 -> WGS84 / UTM zone 36S (southern hemisphere)
    EPSG:32620 -> WGS84 / UTM zone 20N (northern hemisphere)
No external geospatial library is required (pure math, <1m accuracy).

Usage:
    geo_lookup.py                  # all tiles -> compact list + CSV export
    geo_lookup.py chimanimani      # only tiles whose name contains the term
    geo_lookup.py usa_puertorico_s2_999.nc
"""
import os
import sys
import glob
import warnings
import math
from netCDF4 import Dataset

warnings.filterwarnings("ignore")

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
K0 = 0.9996
CSV_PATH = "./datasets/TrainData/tile_coordinates.csv"


def epsg_to_utm_zone(crs):
    """Parse 'EPSG:32636' / 'EPSG:32736' into (zone, southern_hemisphere)."""
    code = int(crs.split(":")[-1])
    if 32601 <= code <= 32660:
        return code - 32600, False
    if 32701 <= code <= 32760:
        return code - 32700, True
    raise ValueError(f"Unsupported CRS: {crs} (only UTM zones supported)")


def utm_to_latlon(easting, northing, zone, southern):
    """Convert UTM (m) to WGS84 (lat, lon) in decimal degrees."""
    e2 = WGS84_F * (2 - WGS84_F)
    e = math.sqrt(e2)
    ep2 = e2 / (1 - e2)

    x = easting - 500000.0
    y = northing if not southern else northing - 10000000.0
    m = y / K0

    mu = m / (WGS84_A * (1 - e2 / 4 - 3 * e2 ** 2 / 64 - 5 * e2 ** 3 / 256))
    e1 = (1 - math.sqrt(1 - e2)) / (1 + math.sqrt(1 - e2))

    phi1 = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * math.sin(2 * mu) \
        + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * math.sin(4 * mu) \
        + (151 * e1 ** 3 / 96) * math.sin(6 * mu) \
        + (1097 * e1 ** 4 / 512) * math.sin(8 * mu)

    n1 = WGS84_A / math.sqrt(1 - e2 * math.sin(phi1) ** 2)
    t1 = math.tan(phi1) ** 2
    c1 = ep2 * math.cos(phi1) ** 2
    r1 = WGS84_A * (1 - e2) / (1 - e2 * math.sin(phi1) ** 2) ** 1.5
    d = x / (n1 * K0)

    lat = phi1 - (n1 * math.tan(phi1) / r1) * (
        d ** 2 / 2
        - (5 + 3 * t1 + 10 * c1 - 4 * c1 ** 2 - 9 * ep2) * d ** 4 / 24
        + (61 + 90 * t1 + 298 * c1 + 45 * t1 ** 2 - 252 * ep2 - 3 * c1 ** 2) * d ** 6 / 720
    )
    lon0 = zone * 6 - 183  # central meridian of the zone, in degrees
    # The lon0 offset + d-terms/cos(phi1) are already in DEGREES; do NOT
    # convert with math.degrees (only the latitude needs it).
    lon = lon0 + (d - (1 + 2 * t1 + c1) * d ** 3 / 6
                  + (5 - 2 * c1 + 28 * t1 - 3 * c1 ** 2 + 8 * ep2 + 24 * t1 ** 2)
                  * d ** 5 / 120) / math.cos(phi1)
    return math.degrees(lat), lon


def read_tile_coords(nc_path):
    """Return (center_lat, center_lon, bbox_ll, bbox_ur) in WGS84 degrees."""
    with Dataset(nc_path, "r") as src:
        crs = str(src.getncattr("crs"))
        c_lat = float(src.getncattr("center_lat"))   # actually UTM northing
        c_lon = float(src.getncattr("center_lon"))   # actually UTM easting
        bbox = src.getncattr("ann_bbox")
    zone, southern = epsg_to_utm_zone(crs)
    lat, lon = utm_to_latlon(c_lon, c_lat, zone, southern)
    bbox_ll = bbox_ur = None
    if isinstance(bbox, (tuple, list)) and len(bbox) == 4:
        bbox_ll = utm_to_latlon(float(bbox[0]), float(bbox[1]), zone, southern)
        bbox_ur = utm_to_latlon(float(bbox[2]), float(bbox[3]), zone, southern)
    return lat, lon, bbox_ll, bbox_ur, crs


def google_maps_link(lat, lon):
    return f"https://www.google.com/maps?q={lat:.6f},{lon:.6f}"


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else ""
    files = sorted(glob.glob("datasets/s2/**/*.nc", recursive=True))
    if query:
        files = [f for f in files if query.lower() in os.path.basename(f).lower()]
    if not files:
        print("No tiles matched.")
        sys.exit(1)

    rows = []
    print(f"{len(files)} tile(s) found.\n")
    for fp in files:
        name = os.path.basename(fp)
        try:
            lat, lon, bbox_ll, bbox_ur, crs = read_tile_coords(fp)
        except Exception as e:
            print(f"  {name}: FAILED ({e})")
            continue
        link = google_maps_link(lat, lon)
        print(f"  {name}")
        print(f"    center  : {lat:.6f}, {lon:.6f}   ({crs})")
        print(f"    Google  : {link}")
        if bbox_ll and bbox_ur:
            print(f"    bbox    : ({bbox_ll[0]:.6f},{bbox_ll[1]:.6f}) -> ({bbox_ur[0]:.6f},{bbox_ur[1]:.6f})")
        rows.append((name, lat, lon, crs))

    if rows:
        with open(CSV_PATH, "w") as cf:
            cf.write("file,lat,lon,crs\n")
            for name, lat, lon, crs in rows:
                cf.write(f"{name},{lat:.6f},{lon:.6f},{crs}\n")
        print(f"\nCSV export: {CSV_PATH}")
