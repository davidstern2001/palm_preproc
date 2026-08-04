"""Coalescing of user (child-domain) layers with clipped DATA_raw layers.

Refactor of merge.py. The user layer has priority inside the "priority
footprint" (the original domain.shp geometry); the clipped raw layer fills
everything outside it, up to the domain rectangle.

Two modes:
  * "difference" (default): raw geometries are CUT along the priority
    footprint boundary (geometry.difference), so the result has neither gaps
    nor duplicated coverage at the seam. Works for polygons, lines and points
    alike (a point inside the footprint differences to empty).
  * "predicate": whole-feature selection, exactly like the original merge.py
    (user kept by `predicate` against the footprint, raw kept where NOT
    intersecting it).

A standalone CLI (`python -m palm_preproc.merge ...`) replicates the original
merge.py workflow (child trees inside a domain + parent trees outside).
"""

from pathlib import Path

import pandas as pd
from shapely import wkt as shapely_wkt

from ..log import get_logger

log = get_logger()


# ------------------------------
# 1. HELPERS
# ------------------------------
def _read_normalized(path, target_crs):
    """Read a vector layer, lowercase its columns, reproject to target CRS."""
    import geopandas as gpd
    gdf = gpd.read_file(path)
    gdf.columns = gdf.columns.str.lower()
    if gdf.crs is None:
        raise ValueError(f"{path}: no CRS defined.")
    if str(gdf.crs) != str(target_crs):
        gdf = gdf.to_crs(target_crs)
    return gdf


def _column_report(user, raw):
    """Report attribute columns that exist in only one of the two inputs
    (generalisation of the original polokor/polokmen diagnostic)."""
    u = set(user.columns) - {"geometry"}
    r = set(raw.columns) - {"geometry"}
    msgs = []
    if u - r:
        msgs.append(f"only in user layer: {sorted(u - r)}")
    if r - u:
        msgs.append(f"only in raw layer: {sorted(r - u)}")
    return "; ".join(msgs)


def _finalize(gdf, fix_geometry, int_columns, reassign_id):
    """Post-merge cleanup: buffer(0) on polygonal geometry, nullable-Int64
    casting of attribute columns, and sequential unique id regeneration."""
    notes = []
    if fix_geometry:
        poly = gdf.geom_type.isin(["Polygon", "MultiPolygon"])
        if poly.any():
            gdf.loc[poly, "geometry"] = gdf.loc[poly, "geometry"].buffer(0)
            before = len(gdf)
            gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()]
            if len(gdf) != before:
                notes.append(f"buffer(0) dropped {before - len(gdf)} empties")
    if reassign_id:
        col = reassign_id.lower()
        # Create the column when it is absent, don't just renumber an
        # existing one. Configuring reassign_id means "the output must carry
        # a unique 1..N id under this name" - downstream tools rely on it
        # (PALM-GeM looks landcover up by 'lid'), and a source layer that
        # simply never had the column used to produce output without it.
        existed = col in gdf.columns
        gdf[col] = range(1, len(gdf) + 1)
        notes.append(f"'{col}' {'reassigned' if existed else 'created'} "
                     f"1..{len(gdf)}")
    for col in (c.lower() for c in int_columns or ()):
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce").astype("Int64")
    return gdf, notes


# ------------------------------
# 2. COALESCE
# ------------------------------
def coalesce_task(user_src, raw_src, out, priority_wkt, rect_wkt, target_crs,
                  mode="difference", predicate="within",
                  priority_source="domain",
                  fix_geometry=False, int_columns=(), reassign_id=None,
                  mark_user=None, fill_mode="footprint", mask_wkt=None):
    """Merge user + clipped-raw layer with user priority. Returns a report.

    mask_wkt (optional): a per-layer mask polygon (WKT) that OVERRIDES the
      shared domain footprint for THIS layer only. User features are kept
      inside it and raw fills outside it, so a layer can have its own
      keep-user boundary (e.g. mask_trees.shp covering a different area than
      the landcover-derived domain). When None, the shared domain footprint
      (priority_wkt) is used.

    priority_source:
      * "domain"        raw data is excluded from the whole domain.shp
                        footprint (holes in the user coverage stay holes).
      * "user_coverage" raw data is excluded only where the user layer
                        actually has geometry, so raw features fill any holes
                        inside the footprint. Falls back to "domain" for
                        point/line layers (no area to difference against).

    Post-processing (ported from landcover_processing_fixed.py):
      * fix_geometry:   buffer(0) on polygonal geometry after merging.
      * int_columns:    cast these attribute columns to nullable Int64.
      * reassign_id:    regenerate this column as a unique 1..N sequence
                        after merging (avoids id collisions between sources).
    """
    import geopandas as gpd

    user_src, out = Path(user_src), Path(out)
    raw_src = Path(raw_src) if raw_src else None
    prio = shapely_wkt.loads(mask_wkt if mask_wkt else priority_wkt)
    rect = shapely_wkt.loads(rect_wkt)

    user = _read_normalized(user_src, target_crs)

    extra_notes = []
    if mark_user:
        # provenance: 1 = user feature, 0 = raw feature (set after concat)
        user[mark_user.lower()] = 1

    if priority_source == "user_coverage":
        coverage = user.geometry.union_all()
        if getattr(coverage, "area", 0) > 0:
            prio = coverage
    if raw_src is not None and raw_src.exists():
        raw = _read_normalized(raw_src, target_crs)
    else:
        raw = user.head(0).copy()   # empty frame with the user schema

    col_note = _column_report(user, raw)

    # Safety: constrain the user layer to this domain's rectangle.
    user = user[~user.geometry.is_empty]
    # Constrain the user layer: to the per-layer mask when given (the mask is
    # the authoritative keep-user region), otherwise to the domain rectangle.
    user = gpd.clip(user, prio if mask_wkt else rect)
    user = user[~user.geometry.is_empty]

    if mode == "difference":
        user_kept = user
        raw_kept = raw.copy()
        if not raw_kept.empty:
            # How raw is removed where the user has data, set per layer by
            # fill_mode:
            #  "coverage":  cut raw exactly along the user geometry, so raw
            #     fills the true gaps between user features. Right for
            #     wall-to-wall layers (landcover). Needs areal user geometry.
            #  "footprint": drop ALL raw inside the shared user-domain
            #     footprint (prio), so inside the domain the layer is
            #     user-only and raw appears only outside. Right for
            #     buildings/roofs/walls/trees, whose inter-feature gaps are
            #     streets that must stay empty, and for points/lines (which
            #     have no area to cut against). All footprint layers share the
            #     SAME prio, keeping walls/roofs/trees mutually consistent.
            # A per-layer mask always defines an explicit keep-user boundary,
            # so it forces footprint-style removal (drop raw inside the mask)
            # regardless of fill_mode.
            user_geom = user_kept.geometry.union_all()
            areal = float(getattr(user_geom, "area", 0.0)) > 0
            if fill_mode == "coverage" and areal and not mask_wkt:
                # Difference only the raw features the user geometry can
                # actually touch. Differencing EVERY raw feature against one
                # large unioned geometry costs a full-geometry test per
                # feature, which on a city-wide landcover layer is the most
                # expensive operation in the pipeline; the ones outside the
                # user coverage come through unchanged anyway.
                hit = raw_kept.sindex.query(user_geom, predicate="intersects")
                if len(hit):
                    hit_idx = raw_kept.index[hit]
                    cut = raw_kept.loc[hit_idx].geometry.difference(user_geom)
                    raw_kept.loc[hit_idx, "geometry"] = cut
                    raw_kept = raw_kept[~raw_kept.geometry.is_empty]
            else:
                # Same idea: only features intersecting the footprint can
                # be dropped, so the index decides membership directly.
                hit = raw_kept.sindex.query(prio, predicate="intersects")
                if len(hit):
                    raw_kept = raw_kept.drop(index=raw_kept.index[hit])
    elif mode == "predicate":
        pred = getattr(user.geometry, predicate)
        user_kept = user[pred(prio)]
        raw_kept = raw[~raw.geometry.intersects(prio)] if not raw.empty else raw
    else:
        return f"ERROR {out.name}: unknown merge mode '{mode}'."

    combined = pd.concat([user_kept, raw_kept], ignore_index=True)
    final = gpd.GeoDataFrame(combined, crs=target_crs)
    if final.empty:
        return f"WARNING {out.name}: merged result is empty; nothing written."

    if mark_user:
        col = mark_user.lower()
        final[col] = pd.to_numeric(final[col], errors="coerce").fillna(0).astype("Int64")
    final, fin_notes = _finalize(final, fix_geometry, int_columns, reassign_id)
    extra_notes += fin_notes

    out.parent.mkdir(parents=True, exist_ok=True)
    final.to_file(out)
    msg = (f"OK {out.name}: {len(user_kept)} user + {len(raw_kept)} raw "
           f"features (mode={mode}).")
    if extra_notes:
        msg += " " + "; ".join(extra_notes) + "."
    if col_note:
        msg += f"\ncolumns differ: {col_note}"
    return msg


# ------------------------------
# 3. STANDALONE CLI (replacement for merge.py)
# ------------------------------
def _cli():
    import argparse
    import geopandas as gpd

    p = argparse.ArgumentParser(
        description="Coalesce a child-domain layer (priority inside the "
                    "domain) with a parent layer (outside).")
    p.add_argument("domain", help="domain shapefile defining the priority footprint")
    p.add_argument("child_layer", help="layer with priority inside the domain")
    p.add_argument("parent_layer", help="layer filling the area outside the domain")
    p.add_argument("output", help="output shapefile")
    p.add_argument("--mode", choices=["difference", "predicate"], default="difference")
    p.add_argument("--predicate", default="within")
    p.add_argument("--priority-source", choices=["domain", "user_coverage"],
                   default="domain")
    p.add_argument("--fix-geometry", action="store_true",
                   help="buffer(0) on polygonal geometry after merging")
    p.add_argument("--int-columns", nargs="*", default=["lid", "type", "code"])
    p.add_argument("--reassign-id", default=None,
                   help="column to regenerate as unique 1..N (e.g. lid)")
    p.add_argument("--fill-mode", choices=["coverage", "footprint"],
                   default="footprint",
                   help="coverage: cut raw by user geometry (landcover); "
                        "footprint: drop raw inside the domain (buildings/trees)")
    p.add_argument("--mask", default=None,
                   help="per-layer mask polygon shapefile (keep user inside, "
                        "raw outside), overriding the domain footprint")
    a = p.parse_args()

    from ..log import setup_logging
    setup_logging()

    domain = gpd.read_file(a.domain)
    target_crs = str(domain.crs)
    prio = domain.geometry.union_all()
    # Standalone use has no snapped rectangle: use the domain bounds envelope.
    rect = prio.envelope

    msg = coalesce_task(a.child_layer, a.parent_layer, a.output,
                        prio.wkt, rect.wkt, target_crs, a.mode, a.predicate,
                        a.priority_source, a.fix_geometry,
                        a.int_columns, a.reassign_id, None, a.fill_mode,
                        (gpd.read_file(a.mask).to_crs(target_crs)
                         .geometry.union_all().wkt if a.mask else None))
    (log.error if msg.startswith("ERROR") else log.info)(msg)


if __name__ == "__main__":
    _cli()
