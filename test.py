#!/usr/bin/env python3
"""palm_preproc test.

Builds a tiny synthetic dataset in a temporary directory, runs the full
pipeline over it, and checks that the expected outputs appear. Nothing
outside the temporary directory is touched, and no real data is needed.

Run it after installing the dependencies, before pointing the pipeline at
real data:

    python test.py            # run, report, clean up
    python test.py --keep     # keep the temporary directory for inspection

Exit status is 0 on success, 1 on failure, so it also works in CI.
"""

# ------------------------------
# 1. INPUT AND OUTPUT FILES
# ------------------------------
import argparse
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Synthetic domain: a 400 m x 400 m patch in UTM 33N, with a few buildings,
# a river strip, and gently sloping terrain.
# UTM zone 33N. Given as an explicit PROJ definition rather than the bare
# code "EPSG:32633" on purpose: creating data from a bare EPSG code forces
# a lookup in PROJ's proj.db, which fails on environments with a stale or
# mismatched PROJ installation. The pipeline itself only ever READS the
# CRS embedded in existing files, so it is unaffected by that; the
# self-test should not be stricter than the program it tests.
CRS = ("+proj=utm +zone=33 +datum=WGS84 +units=m +no_defs")
ORIGIN_X, ORIGIN_Y = 460000.0, 5548000.0
EXTENT = 400.0
RAW_RES = 2.0                      # raw raster resolution [m]


# ------------------------------
# 2. DEPENDENCY CHECK
# ------------------------------
def check_dependencies():
    """Import every third-party dependency and report versions, so a broken
    environment fails here with a clear message instead of deep inside a
    stage."""
    missing, versions = [], {}
    for mod, label in (("geopandas", "geopandas"), ("shapely", "shapely"),
                       ("rasterio", "rasterio"), ("yaml", "pyyaml"),
                       ("numpy", "numpy"), ("pandas", "pandas")):
        try:
            m = __import__(mod)
            versions[label] = getattr(m, "__version__", "?")
        except ImportError as exc:
            missing.append(f"{label} ({exc})")
    if missing:
        print("MISSING DEPENDENCIES:")
        for m in missing:
            print(f"  - {m}")
        print("\nInstall them with:  pip install -r requirements.txt")
        return False
    print("Dependencies:")
    for k, v in versions.items():
        print(f"  {k:<12} {v}")

    # PROJ database health. A stale/mismatched proj.db breaks EPSG *code*
    # lookups while leaving explicit CRS definitions (and the CRS already
    # embedded in existing data files) working. The pipeline reads CRS from
    # its inputs, so it still runs; only features that resolve a bare EPSG
    # code - notably crs.domain_output reprojection - would fail.
    try:
        import rasterio.crs
        rasterio.crs.CRS.from_epsg(32633)
        print("  PROJ database  OK (EPSG lookups work)")
    except Exception as exc:
        first = str(exc).split(chr(10))[0]
        print("\n  WARNING: EPSG code lookups fail in this environment:")
        print(f"    {first}")
        print("    The pipeline itself is unaffected (it reads the CRS stored")
        print("    in your data files), but crs.domain_output reprojection")
        print("    would not work. Usually caused by a stale PROJ_DATA/PROJ_LIB")
        print("    variable or by mixing conda and pip GDAL/PROJ packages.")
        print("    Check:  echo $PROJ_DATA $PROJ_LIB   and")
        print("            python -c 'import pyproj; print(pyproj.datadir.get_data_dir())'")
    return True


# ------------------------------
# 3. SYNTHETIC DATA GENERATION
# ------------------------------
def make_raw_data(raw_dir):
    """Whole-'city' source data: DEM, building heights, and the four vector
    layers, covering a larger area than the domain of interest."""
    import numpy as np
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box, Point
    from shapely.ops import unary_union

    raw_dir.mkdir(parents=True, exist_ok=True)
    n = int(EXTENT / RAW_RES)
    transform = from_origin(ORIGIN_X, ORIGIN_Y + EXTENT, RAW_RES, RAW_RES)

    # DEM: a gentle slope plus a bare hill (tests that a hill with no
    # buildings does not inflate the child nz).
    yy, xx = np.mgrid[0:n, 0:n]
    dem = 200.0 + 0.02 * xx + 0.01 * yy
    dem[10:30, 10:30] += 25.0                      # bare hill
    _write_raster(raw_dir / "dem.tif", dem, transform, nodata=-32768.0)

    # Buildings: heights above terrain, nodata elsewhere.
    bld = np.full((n, n), -9999.0)
    for (r0, r1, c0, c1, hgt) in ((60, 75, 60, 80, 12.0),
                                  (60, 75, 95, 115, 18.0),
                                  (95, 110, 60, 80, 9.0),
                                  (95, 115, 100, 120, 24.0)):
        bld[r0:r1, c0:c1] = hgt
    _write_raster(raw_dir / "buildings.tif", bld, transform, nodata=-9999.0)

    # Vector layers. Roof polygons match the building blocks above.
    roofs, walls = [], []
    for i, (r0, r1, c0, c1) in enumerate(((60, 75, 60, 80), (60, 75, 95, 115),
                                          (95, 110, 60, 80), (95, 115, 100, 120))):
        x0 = ORIGIN_X + c0 * RAW_RES
        x1 = ORIGIN_X + c1 * RAW_RES
        y1 = ORIGIN_Y + EXTENT - r0 * RAW_RES
        y0 = ORIGIN_Y + EXTENT - r1 * RAW_RES
        poly = box(x0, y0, x1, y1)
        roofs.append({"bid": i + 1, "katroof": 1, "geometry": poly})
        walls.append({"bid": i + 1, "katwall": 1,
                      "geometry": poly.buffer(1.0).difference(poly)})
    gpd.GeoDataFrame(roofs, crs=CRS).to_file(raw_dir / "roofs.shp")
    gpd.GeoDataFrame(walls, crs=CRS).to_file(raw_dir / "walls.shp")

    # Landcover: a grass background tiled into parcels, a paved strip and a
    # river strip, plus 'building' parcels under the roofs.
    # Grass parcels tiled over the whole area, then water and building
    # parcels punched into them, so the layer stays a clean partition with
    # no overlaps (overlapping parcels would make the clip seam ambiguous).
    lc = []
    lid = 1
    # 37 m tiles offset by 7 m: deliberately NOT aligned with the domain
    # rectangle, so clipping never produces zero-area edge slivers
    # (degenerate line/point geometries cannot be written to a polygon
    # shapefile).
    step = 37.0
    off = 7.0
    water = box(ORIGIN_X, ORIGIN_Y + 330.0,
                ORIGIN_X + EXTENT, ORIGIN_Y + 370.0)
    building_union = unary_union([r["geometry"] for r in roofs])
    ny_ = int(EXTENT / step) + 2
    for iy in range(ny_):
        for ix in range(ny_):
            x = ORIGIN_X - off + ix * step
            y = ORIGIN_Y - off + iy * step
            g = box(x, y, x + step, y + step).difference(water)
            g = g.difference(building_union)
            if g.is_empty or g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            lc.append({"lid": lid, "code": 3, "type": 3, "geometry": g})
            lid += 1
    lc.append({"lid": lid, "code": 2, "type": 2, "geometry": water})
    lid += 1
    for r in roofs:                                     # building parcels
        lc.append({"lid": lid, "code": 7, "type": 7, "geometry": r["geometry"]})
        lid += 1
    gdf = gpd.GeoDataFrame(lc, crs=CRS)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
    gdf.to_file(raw_dir / "landcover.shp")

    # Trees: a handful of crowns.
    trees = [{"tid": i + 1, "height": 8.0 + i,
              "geometry": Point(ORIGIN_X + 120.0 + 25.0 * i,
                                ORIGIN_Y + 150.0).buffer(4.0)}
             for i in range(5)]
    gpd.GeoDataFrame(trees, crs=CRS).to_file(raw_dir / "trees.shp")

    (raw_dir / "surface_params.csv").write_text("id,param\n1,0.1\n")


def make_user_data(user_dir):
    """Project-specific input: here only the area of interest, so the run
    exercises the domain_from path and the clip/merge stages on raw data."""
    import geopandas as gpd
    from shapely.geometry import box

    user_dir.mkdir(parents=True, exist_ok=True)
    aoi = box(ORIGIN_X + 100.0, ORIGIN_Y + 100.0,
              ORIGIN_X + 260.0, ORIGIN_Y + 260.0)
    gpd.GeoDataFrame([{"id": 1, "geometry": aoi}], crs=CRS).to_file(
        user_dir / "domain.shp")


def _write_raster(path, arr, transform, nodata):
    import rasterio
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0],
                       width=arr.shape[1], count=1, dtype="float32",
                       crs=CRS, transform=transform, nodata=nodata) as ds:
        ds.write(arr.astype("float32"), 1)


def make_config(root, cfg_path):
    """A minimal project config: everything else comes from the site
    defaults, which is exactly what a real project should look like."""
    cfg_path.write_text(f"""project:
  name: test
  root: {root}
  output_dir: ./test
  overwrite: true

user_data:
  dir: ./test/DATA_user
  domain: domain.shp

domains:
  child:
    grid_size: 2.0
    confirm_optimized: false
  parent:
    grid_size: 4.0
    buffer: 40.0

# Single-worker: the self-test should run anywhere, including memory-limited
# CI containers, and the dataset is far too small for parallelism to help.
clip:
  workers: 1
merge:
  workers: 1

templates:
  values:
    origin_time: "2023-08-23 19:00:00"
    length: 2
    wrf_date: "2023-08-23"
    hpc_user: test
    wrf_dir: /tmp/wrf
    node_cpus: 8
    min_nodes: 1
    max_nodes: 8
""")


# ------------------------------
# 4. RUN AND CHECK
# ------------------------------
EXPECTED = [
    "domain_child.shp", "domain_parent.shp", "domains_report.txt",
    "test_p3d", "test_p3d_N02", "test_p3dr", "test_p3dr_N02",
    "pgem_test.yaml", "pgem_test_N02.yaml",
    "pmeteo_test.yaml", "pmeteo_test_N02.yaml",
    "DATA_child/dem.tif", "DATA_child/buildings.tif",
    "DATA_child/landcover.shp", "DATA_child/roofs.shp",
    "DATA_parent/dem.tif", "DATA_parent/landcover.shp",
]


def run(keep=False):
    if not check_dependencies():
        return 1

    tmp = Path(tempfile.mkdtemp(prefix="palm_preproc_test_"))
    print(f"\nWorking directory: {tmp}")
    try:
        print("Generating synthetic data ...")
        make_raw_data(tmp / "DATA_raw")
        make_user_data(tmp / "test" / "DATA_user")
        cfg_path = tmp / "test.yaml"
        make_config(tmp, cfg_path)

        print("Running the pipeline ...\n" + "-" * 70)
        sys.path.insert(0, str(HERE))
        from palm_preproc.pipeline import main
        rc = main(["-c", str(cfg_path)])
        print("-" * 70)
        if rc != 0:
            print(f"FAILED: pipeline exited with status {rc}")
            return 1

        out = tmp / "test"
        missing = [f for f in EXPECTED if not (out / f).exists()]
        if missing:
            print("FAILED: expected outputs missing:")
            for f in missing:
                print(f"  - {f}")
            return 1

        # The generated namelist should carry real values, not placeholders.
        p3d = (out / "test_p3d").read_text()
        leftovers = [tok for tok in ("<hpc_user>", "<wrf_dir>", "<time")
                     if tok in p3d]
        pgem = (out / "pgem_test.yaml").read_text()
        if "<hpc_user>" in pgem:
            leftovers.append("<hpc_user> (pgem)")
        if leftovers:
            print(f"FAILED: unfilled placeholders in output: {leftovers}")
            return 1

        print(f"\nOK - {len(EXPECTED)} expected outputs present, "
              f"templates filled.")
        print("palm_preproc works in this environment.")
        return 0
    except Exception:
        print("FAILED with an exception:\n")
        traceback.print_exc()
        return 1
    finally:
        if keep:
            print(f"\nKept: {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="palm_preproc self-test")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temporary directory for inspection")
    sys.exit(run(ap.parse_args().keep))
