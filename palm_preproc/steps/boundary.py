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

# ------------------------------
# 1. LANDCOVER RELABELING
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

    # Every candidate lookup goes through the spatial index. Testing each
    # removed building against the WHOLE landcover frame (two full
    # .intersection().area passes per building) is O(n_removed x
    # n_landcover) and dominated the stage: a domain with a few hundred
    # boundary buildings and a real city landcover layer spent minutes here.
    # The index reduces each building to the handful of parcels it can
    # actually touch; the arithmetic below is unchanged.
    sindex = landcover.sindex
    lc_geom = landcover.geometry
    lc_area = lc_geom.area

    for fp in geoms:
        if fp is None or fp.is_empty or fp.area <= 0:
            continue

        # -- parcels under THIS building -------------------------------
        cand = sindex.query(fp, predicate="intersects")
        if len(cand) == 0:
            continue
        cand_idx = landcover.index[cand]
        sub = lc_geom.iloc[cand]
        inter = sub.intersection(fp).area
        # local threshold: fraction of the smaller of parcel / footprint
        denom = lc_area.iloc[cand].clip(upper=fp.area)
        frac = inter / denom.replace(0, pd.NA)
        under_mask = (frac > min_overlap_frac).fillna(False)
        under_idx = cand_idx[under_mask.values]
        # A parcel already relabeled by an earlier building keeps that
        # label: the docstring has always said the first claimant wins,
        # but the previous code reassigned it on every later hit, so the
        # LAST building actually won. Now it matches what is documented.
        under_idx = under_idx.difference(pd.Index(sorted(relabeled)))
        if under_idx.empty:
            continue

        # -- majority vote from the ring around THIS building ----------
        ring = fp.buffer(buffer_m).difference(all_removed)
        if ring.is_empty:
            continue
        rcand = sindex.query(ring, predicate="intersects")
        if len(rcand) == 0:
            continue
        rcand_idx = landcover.index[rcand]
        keep = ~rcand_idx.isin(under_idx)
        if not keep.any():
            continue
        rcand_idx = rcand_idx[keep]
        ring_geom = lc_geom.loc[rcand_idx]
        # One intersection pass, reused as both the surround test and the
        # vote weights (it used to be computed twice).
        weights = ring_geom.intersection(ring).area
        pos = weights.values > 0
        if not pos.any():
            continue
        surround_idx = rcand_idx[pos]
        w = weights.values[pos]

        for col in cols:
            tmp = pd.DataFrame({"val": landcover.loc[surround_idx, col].values,
                                "w": w}).dropna(subset=["val"])
            tmp = tmp[tmp["w"] > 0]
            if tmp.empty:
                continue
            majority = tmp.groupby("val")["w"].sum().idxmax()
            landcover.loc[under_idx, col] = majority
        relabeled.update(under_idx)
    return len(relabeled)


def _walls_under(walls, removed_fp, kept_fp):
    """Boolean mask of the walls (a GeoSeries) that belong to removed roofs.

    Polygon walls: representative point (interior) inside the removed
    footprint, or most of their area inside it - so a wall merely sharing
    an edge with a removed roof stays.

    Line and point walls: a wall drawn ON a roof outline lies on the
    footprint's boundary, where `within` is false, so the representative-
    point test never matched them and every line wall of a removed building
    survived. They are matched by how much of them the removed footprint
    covers (boundary included) instead; one covered just as much by a KEPT
    roof sits on an edge two buildings share and is kept with that one.
    """
    reps = walls.representative_point()
    wmask = reps.within(removed_fp)
    poly = walls.geom_type.isin(["Polygon", "MultiPolygon"])
    if poly.any():
        inside_area = walls[poly].intersection(removed_fp).area
        frac = inside_area / walls[poly].area.replace(0, pd.NA)
        wmask[poly] = wmask[poly] | (frac > 0.5).fillna(False)
    lines = walls.geom_type.isin(["LineString", "MultiLineString",
                                  "LinearRing"])
    if lines.any():
        length = walls[lines].length.replace(0, pd.NA)
        on_removed = (walls[lines].intersection(removed_fp).length
                      / length > 0.5).fillna(False)
        on_kept = (walls[lines].intersection(kept_fp).length
                   / length > 0.5).fillna(False)
        wmask[lines] = wmask[lines] | (on_removed & ~on_kept)
    points = walls.geom_type.isin(["Point", "MultiPoint"])
    if points.any():
        wmask[points] = (walls[points].covered_by(removed_fp)
                         & ~walls[points].covered_by(kept_fp))
    return wmask.astype(bool)


def _union(geoms):
    from shapely.ops import unary_union
    return unary_union(list(geoms))


# ------------------------------
# 2. PER-DOMAIN CLEANUP
# ------------------------------
def clean_boundary(domain_name, cfg, spec):
    """Remove boundary-truncated buildings from one domain's final layers
    (roofs, walls, landcover, buildings raster). Returns a report string."""
    ddir = cfg.data_dir(domain_name)
    # Written filenames: vectors are always .shp regardless of input format.
    roofs_path = ddir / cfg.output_layer_filename("roofs")
    walls_path = ddir / cfg.output_layer_filename("walls")
    landcover_path = ddir / cfg.output_layer_filename("landcover")
    buildings_path = ddir / cfg.output_layer_filename("buildings")
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
            wmask = _walls_under(walls.geometry, removed_fp,
                                 kept.geometry.union_all())
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
