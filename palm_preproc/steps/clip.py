"""Clipping of DATA_raw layers to the child/parent domain rectangles.

Vector layers are reprojected to the aligned CRS and clipped to the domain
rectangle. Raster layers are (by default) resampled onto the EXACT PALM grid
of the domain -- same origin, same grid_size, width_pts x height_pts cells --
so the static driver can consume them without any further alignment.

Worker functions take only picklable arguments (paths, WKT, plain dicts) so
they run cleanly under ProcessPoolExecutor.
"""

from pathlib import Path

import numpy as np
from shapely import wkt as shapely_wkt

from .domains import DomainSpec
from ..log import get_logger

log = get_logger()


# ------------------------------
# 1. VECTOR CLIP
# ------------------------------
def _clean_after_clip(gdf):
    """Drop empties; explode GeometryCollections into their parts."""
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if (gdf.geom_type == "GeometryCollection").any():
        gdf = gdf.explode(index_parts=False)
        gdf = gdf[~gdf.geometry.is_empty]
        gdf = gdf[gdf.geom_type != "GeometryCollection"]
    return gdf


def clip_vector_task(src, out, rect_wkt, target_crs):
    """Clip one vector layer to the domain rectangle. Returns a report string."""
    import geopandas as gpd

    src, out = Path(src), Path(out)
    rect = shapely_wkt.loads(rect_wkt)

    gdf = gpd.read_file(src)
    if gdf.crs is None:
        return f"ERROR {src.name}: no CRS defined on input."
    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)

    clipped = _clean_after_clip(gpd.clip(gdf, rect))
    if clipped.empty:
        return (f"WARNING {src.name}: no features inside the domain; "
                f"nothing written to {out.name}.")

    out.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_file(out)
    return f"OK {out.name}: {len(clipped)}/{len(gdf)} features."


# ------------------------------
# 2. RASTER CLIP / GRID SNAP
# ------------------------------
def _proj_db_layout(path):
    """(MAJOR, MINOR) database layout version of a proj.db, read directly
    via sqlite so PROJ itself never touches an incompatible file (which
    would print C-level ERROR lines to stderr). None if unreadable."""
    import sqlite3
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = con.execute(
            "SELECT key, value FROM metadata WHERE key IN "
            "('DATABASE.LAYOUT.VERSION.MAJOR', 'DATABASE.LAYOUT.VERSION.MINOR')"
        ).fetchall()
        con.close()
        d = dict(rows)
        return (int(d["DATABASE.LAYOUT.VERSION.MAJOR"]),
                int(d["DATABASE.LAYOUT.VERSION.MINOR"]))
    except Exception:
        return None


def _suppressed_stderr():
    """Context manager silencing fd-level stderr (GDAL/PROJ C output)."""
    import contextlib
    import os

    @contextlib.contextmanager
    def cm():
        saved = os.dup(2)
        try:
            devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull, 2)
            os.close(devnull)
            yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)
    return cm()


_PROJ_NOTE = ("\nPROJ_LIB/PROJ_DATA incompatible; bundled PROJ data used "
              "(permanent fix: conda-forge rasterio, or unset the variables)")


def _use_bundled_proj(bundled):
    import os
    os.environ["PROJ_LIB"] = str(bundled)
    os.environ["PROJ_DATA"] = str(bundled)
    try:
        from rasterio._env import set_proj_data_search_path
        set_proj_data_search_path(str(bundled))
    except Exception:
        pass


def _import_rasterio():
    """Import rasterio, working around PROJ database conflicts.

    In conda environments, PROJ_LIB/PROJ_DATA may point at the environment's
    proj.db while a pip-installed rasterio wheel bundles its own (newer)
    PROJ, which then rejects that database ("DATABASE.LAYOUT.VERSION...
    comes from another PROJ installation"). The incompatibility is detected
    PREEMPTIVELY by comparing the database layout versions via sqlite, so
    PROJ never opens the bad file and never prints C-level ERROR lines; a
    silenced probe remains as a fallback for other breakages.
    """
    import os
    import rasterio
    from rasterio.crs import CRS

    note = ""
    bundled = Path(rasterio.__file__).parent / "proj_data"
    env_dir = os.environ.get("PROJ_DATA") or os.environ.get("PROJ_LIB")
    if bundled.is_dir() and env_dir and Path(env_dir) != bundled:
        env_layout = _proj_db_layout(Path(env_dir) / "proj.db")
        bun_layout = _proj_db_layout(bundled / "proj.db")
        if bun_layout and (env_layout is None or env_layout < bun_layout):
            _use_bundled_proj(bundled)
            note = _PROJ_NOTE

    try:
        with _suppressed_stderr():
            CRS.from_epsg(4326)
        return rasterio, note
    except Exception:
        if not bundled.is_dir():
            raise
        _use_bundled_proj(bundled)
        with _suppressed_stderr():
            CRS.from_epsg(4326)   # raises again if still broken
        return rasterio, _PROJ_NOTE


def _default_nodata(dtype):
    if np.issubdtype(dtype, np.floating):
        return -9999.0
    if np.issubdtype(dtype, np.signedinteger):
        return -9999 if np.iinfo(dtype).min <= -9999 else np.iinfo(dtype).min
    return np.iinfo(dtype).max  # unsigned: use the max value


def clip_raster_task(src, out, spec_dict, resampling_name, snap_to_grid,
                     force_nodata=None):
    """Clip (and optionally grid-snap) one raster. Returns a report string."""
    rasterio, proj_note = _import_rasterio()
    from rasterio.mask import mask as rio_mask
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    src, out = Path(src), Path(out)
    spec = DomainSpec.from_dict(spec_dict)
    out.parent.mkdir(parents=True, exist_ok=True)
    resampling = Resampling[resampling_name]

    with rasterio.open(src) as ds:
        dtype = ds.dtypes[0]
        nodata = ds.nodata
        note = ""
        if force_nodata is not None:
            nodata = force_nodata
            # a float nodata like -9999.0 needs a float raster
            if not np.issubdtype(np.dtype(dtype), np.floating):
                dtype = "float32"
            note = f" (nodata forced to {nodata})"
        elif nodata is None:
            nodata = _default_nodata(np.dtype(ds.dtypes[0]))
            note = f" (no src nodata; using {nodata})"

        if snap_to_grid:
            # Resample onto the exact PALM grid: origin + grid_size + pts.
            dst_transform = from_origin(spec.origin_x, spec.maxy,
                                        spec.grid_size, spec.grid_size)
            dst = np.full((ds.count, spec.height_pts, spec.width_pts),
                          nodata, dtype=dtype)
            for b in range(1, ds.count + 1):
                reproject(
                    source=rasterio.band(ds, b),
                    destination=dst[b - 1],
                    dst_transform=dst_transform,
                    dst_crs=spec.crs,
                    src_nodata=ds.nodata,
                    dst_nodata=nodata,
                    resampling=resampling,
                )
            profile = ds.profile.copy()
            profile.update(
                crs=spec.crs, transform=dst_transform, nodata=nodata,
                width=spec.width_pts, height=spec.height_pts,
                driver="GTiff", compress="deflate", dtype=dtype,
            )
            for k in ("tiled", "blockxsize", "blockysize"):
                profile.pop(k, None)
            with rasterio.open(out, "w", **profile) as dst_ds:
                dst_ds.write(dst)
            valid = int(np.count_nonzero(dst[0] != nodata))
            total = spec.width_pts * spec.height_pts
            extra = f"; {100*valid/total:.1f}% valid cells" if total else ""
            return (f"OK {out.name}: {spec.width_pts}x{spec.height_pts} @ "
                    f"{spec.grid_size:g} m, {resampling_name}{note}{extra}"
                    f"{proj_note}")

        # Plain crop in the raster's native CRS (no grid snapping).
        import geopandas as gpd
        rect = gpd.GeoSeries([spec.rectangle()], crs=spec.crs).to_crs(ds.crs)
        data, transform = rio_mask(ds, rect.geometry, crop=True, nodata=nodata)
        profile = ds.profile.copy()
        profile.update(transform=transform, nodata=nodata, driver="GTiff",
                       width=data.shape[2], height=data.shape[1])
        for k in ("tiled", "blockxsize", "blockysize"):
            profile.pop(k, None)
        with rasterio.open(out, "w", **profile) as dst_ds:
            dst_ds.write(data)
        return f"OK {out.name}: cropped in native CRS ({ds.crs}){note}{proj_note}"


# ------------------------------
# 3. BUILDINGS MASK (clip the buildings raster to the roofs layer)
# ------------------------------
def mask_raster_to_vector_task(raster_path, vector_path, nodata):
    """Set raster cells outside the vector layer's geometry to nodata.

    Used to constrain buildings.tif to the FINAL (possibly user-merged)
    roofs.shp of the same domain, in place. Returns a report string."""
    import geopandas as gpd
    from rasterio.features import geometry_mask

    rasterio, proj_note = _import_rasterio()
    raster_path, vector_path = Path(raster_path), Path(vector_path)

    gdf = gpd.read_file(vector_path)
    with rasterio.open(raster_path) as ds:
        data = ds.read()
        profile = ds.profile.copy()
        if str(gdf.crs) != str(ds.crs):
            gdf = gdf.to_crs(ds.crs)
        if gdf.empty:
            return (f"WARNING {raster_path.name}: {vector_path.name} has no "
                    f"features; mask skipped.")
        inside = geometry_mask(gdf.geometry, out_shape=data.shape[1:],
                               transform=ds.transform, invert=True,
                               all_touched=True)
    dtype = data.dtype
    if not np.issubdtype(dtype, np.floating):
        data = data.astype("float32")
        dtype = np.dtype("float32")
    data[:, ~inside] = nodata
    # GDAL >= 3.10 may open TIFFs via the read-only LIBERTIFF driver and
    # report it in the profile; writing must always go through GTiff.
    profile.update(nodata=nodata, dtype=str(dtype), driver="GTiff")
    for k in ("tiled", "blockxsize", "blockysize"):
        profile.pop(k, None)
    with rasterio.open(raster_path, "w", **profile) as out:
        out.write(data)
    frac = 100.0 * inside.sum() / inside.size
    return (f"OK {raster_path.name}: masked to {vector_path.name} "
            f"({frac:.1f}% of cells inside roofs; nodata {nodata})"
            f"{proj_note}")


# ------------------------------
# 4. RASTER MERGE (user raster over clipped raw raster)
# ------------------------------
def merge_raster_task(user_src, raw_src, out, spec_dict, priority_wkt,
                      resampling_name, force_nodata):
    """Composite a user raster over the clipped raw raster on the domain's
    exact PALM grid: inside the priority footprint every cell is taken from
    the user raster (resampled), outside it from the raw raster. Cells that
    would be user-driven but where the user raster has nodata fall back to
    raw so no holes are introduced. Returns a report string."""
    import geopandas as gpd
    import numpy as np
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    from shapely import wkt as shapely_wkt

    rasterio, proj_note = _import_rasterio()
    user_src, raw_src, out = Path(user_src), Path(raw_src), Path(out)
    spec = DomainSpec.from_dict(spec_dict)
    resampling = Resampling[resampling_name]
    prio = shapely_wkt.loads(priority_wkt)

    dst_transform = from_origin(spec.origin_x, spec.maxy,
                                spec.grid_size, spec.grid_size)
    shape = (spec.height_pts, spec.width_pts)

    # base = clipped raw raster (already on the PALM grid from the clip stage)
    with rasterio.open(raw_src) as rds:
        base = rds.read(1).astype("float32")
        profile = rds.profile.copy()
        nodata = force_nodata if force_nodata is not None else rds.nodata
        if nodata is None:
            nodata = _default_nodata(np.dtype("float32"))

    # user raster -> resample onto the exact PALM grid
    with rasterio.open(user_src) as uds:
        user = np.full(shape, np.nan, dtype="float32")
        reproject(
            source=rasterio.band(uds, 1), destination=user,
            dst_transform=dst_transform, dst_crs=spec.crs,
            src_nodata=uds.nodata, dst_nodata=np.nan, resampling=resampling,
        )
        if uds.nodata is not None:
            user[user == uds.nodata] = np.nan

    # priority footprint on the PALM grid (True = inside -> user wins)
    inside = geometry_mask([prio], out_shape=shape, transform=dst_transform,
                           invert=True, all_touched=True)

    # inside & user-valid -> user; everywhere else -> raw
    take_user = inside & np.isfinite(user)
    result = np.where(take_user, user, base).astype("float32")
    result[~np.isfinite(result)] = nodata

    profile.update(driver="GTiff", dtype="float32", nodata=nodata,
                   transform=dst_transform, crs=spec.crs,
                   width=spec.width_pts, height=spec.height_pts,
                   compress="deflate")
    for k in ("tiled", "blockxsize", "blockysize"):
        profile.pop(k, None)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(result, 1)

    n_user = int(take_user.sum())
    frac = 100.0 * n_user / take_user.size if take_user.size else 0.0
    return (f"OK {out.name}: user raster over raw ({frac:.1f}% of cells from "
            f"user inside the footprint){proj_note}")
