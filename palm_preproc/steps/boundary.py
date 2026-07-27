"""Domain-boundary cleanup: remove buildings truncated at the domain edge.

When roofs/walls/landcover are clipped to a domain rectangle, buildings that
straddle the boundary end up as partial polygons -- physically wrong for
PALM (a sliced building sitting at the domain edge). This step, run on the
FINAL per-domain layers (after clip and merge):

  1. Deletes roof polygons that touch the domain rectangle boundary (they
     were truncated by clipping), rather than keeping the partial sliver.
  2. Deletes the corresponding wall polygons, matched by geometry: a wall
     is removed when it lies under a removed roof's footprint. Geometric
     matching works for any input data, independent of which (if any)
     building-id columns the roofs and walls layers carry.
  3. Relabels the landcover polygon(s) that sat under the deleted building
     footprint to the majority code/type of the surrounding landcover (an
     area-weighted vote in a buffer ring around the footprint), so no
     orphaned "building" landcover is left where the building was removed.
  4. Re-masks buildings.tif to the cleaned roofs.shp, so building heights
     are removed for the deleted boundary buildings too.

The boundary_cleanup.roofs_id_column / walls_id_column config keys are
accepted for backward compatibility but have no effect; wall matching is
purely geometric.
"""

import geopandas as gpd
import pandas as pd

from ..log import get_logger

log = get_logger()

# Candidate building-id column names, in priority order.

# ------------------------------
# 1. ID COLUMN DETECTION
# ------------------------------
# ------------------------------
# 2. LANDCOVER RELABELING
# ------------------------------
def _relabel_landcover(landcover, footprints, buffer_m, columns,
                       min_overlap_frac=0.5):
    """Relabel landcover polygons sitting under removed-building footprints to
    the area-weighted majority landcover of each building's immediate
    surroundings. `footprints` is an iterable of individual removed-roof
    geometries (NOT their union). Returns the number of landcover polygons
    relabeled.

    Each removed building is handled INDEPENDENTLY:
      * "under" = landcover polygons whose intersection with THIS footprint
        exceeds min_overlap_frac of the *smaller* of (the landcover polygon,
        the footprint). The threshold is local to each building, so it
        adapts to parcel and footprint size no matter how many buildings
        are removed in total.
      * the majority vote is taken from a ring around THIS footprint only, so
        each building is relabeled from its own local surroundings rather than
        one global winner assigned to every building. The ring excludes all
        removed footprints so an adjacent removed building can't pollute the
        vote.

    A landcover polygon under several removed buildings is relabeled once (the
    first building that claims it wins); it is counted once.
    """
    cols = [c for c in columns if c in landcover.columns]
    if not cols:
        return 0

    all_removed = footprints.union_all() if hasattr(footprints, "union_all") \
        else _union(footprints)
    relabeled = set()

    geoms = footprints.geometry if hasattr(footprints, "geometry") \
        else footprints
    for fp in geoms:
        if fp is None or fp.is_empty or fp.area <= 0:
            continue
        inter = landcover.geometry.intersection(fp).area
        # local threshold: fraction of the smaller of parcel / footprint
        denom = landcover.geometry.area.clip(upper=fp.area)
        frac = inter / denom.replace(0, pd.NA)
        under = landcover[(frac > min_overlap_frac).fillna(False)]
        if under.empty:
            continue

        ring = fp.buffer(buffer_m).difference(all_removed)
        if ring.is_empty:
            continue
        ring_overlap = landcover.geometry.intersection(ring).area
        surround = landcover[(ring_overlap > 0)
                             & ~landcover.index.isin(under.index)]
        if surround.empty:
            continue
        weights = surround.geometry.intersection(ring).area
        for col in cols:
            tmp = pd.DataFrame({"val": surround[col].values,
                                "w": weights.values}).dropna(subset=["val"])
            tmp = tmp[tmp["w"] > 0]
            if tmp.empty:
                continue
            majority = tmp.groupby("val")["w"].sum().idxmax()
            landcover.loc[under.index, col] = majority
        relabeled.update(under.index)
    return len(relabeled)


def _union(geoms):
    from shapely.ops import unary_union
    return unary_union(list(geoms))


# ------------------------------
# 3. PER-DOMAIN CLEANUP
# ------------------------------
def clean_boundary(domain_name, cfg, spec):
    """Remove boundary-truncated buildings from one domain's final layers
    (roofs, walls, landcover, buildings raster). Returns a report string."""
    ddir = cfg.data_dir(domain_name)
    layers = cfg["raw_data"]["layers"]
    roofs_path = ddir / layers["roofs"]
    walls_path = ddir / layers["walls"]
    landcover_path = ddir / layers["landcover"]
    buildings_path = ddir / layers["buildings"]
    bcfg = cfg["boundary_cleanup"]

    if not roofs_path.exists():
        return f"{domain_name}: no roofs layer, skipped."

    roofs = gpd.read_file(roofs_path)
    if roofs.empty:
        return f"{domain_name}: roofs layer empty, nothing to clean."

    rect = spec.rectangle()
    touches = roofs.geometry.intersects(rect.boundary)
    n_removed = int(touches.sum())
    if n_removed == 0:
        return f"{domain_name}: no boundary-truncated roofs."

    removed = roofs[touches]
    kept = roofs[~touches]

    kept.to_file(roofs_path)
    notes = [f"removed {n_removed} boundary roof(s)"]

    # Footprint of the removed (boundary-truncated) roofs. Walls are matched
    # to it by geometry: a wall belongs to a removed roof iff it actually
    # lies under that roof's footprint. This is independent of any
    # building-id columns, so it works for raw and user data alike.
    removed_fp = removed.geometry.union_all()

    # -- walls --------------------------------------------------------
    n_walls = 0
    if walls_path.exists() and not removed.empty:
        walls = gpd.read_file(walls_path)
        if not walls.empty:
            # Test each wall by its representative point (guaranteed on the
            # geometry, interior for polygons) so walls merely sharing an edge
            # with a neighbouring removed footprint are not swept up, while a
            # wall genuinely inside a removed roof is caught regardless of
            # geometry type (polygon perimeter, line or point walls).
            reps = walls.geometry.representative_point()
            wmask = reps.within(removed_fp)
            # Polygon walls that straddle in/out: also drop those whose
            # majority area lies inside the removed footprint.
            poly = walls.geom_type.isin(["Polygon", "MultiPolygon"])
            if poly.any():
                inside_area = walls.loc[poly].geometry.intersection(
                    removed_fp).area
                frac = inside_area / walls.loc[poly].geometry.area.replace(
                    0, pd.NA)
                wmask.loc[poly] = wmask.loc[poly] | (frac > 0.5).fillna(False)
            n_walls = int(wmask.sum())
            if n_walls:
                walls[~wmask].to_file(walls_path)
    notes.append(f"removed {n_walls} wall(s)")

    # -- landcover relabeling ------------------------------------------
    n_relabel = 0
    if landcover_path.exists():
        landcover = gpd.read_file(landcover_path)
        n_relabel = _relabel_landcover(
            landcover, removed, float(bcfg.get("buffer", 20.0)),
            bcfg.get("landcover_columns", ["code", "type"]))
        if n_relabel:
            landcover.to_file(landcover_path)
    notes.append(f"relabeled {n_relabel} landcover polygon(s)")

    # -- re-mask buildings.tif to the cleaned roofs ---------------------
    if (buildings_path.exists()
            and cfg["clip"].get("buildings_mask_layer") == "roofs"):
        from .clip import mask_raster_to_vector_task
        nodata = cfg["clip"].get("buildings_nodata", -9999.0)
        mask_raster_to_vector_task(str(buildings_path), str(roofs_path), nodata)
        notes.append("buildings raster re-masked")

    return f"{domain_name}: " + "; ".join(notes)
