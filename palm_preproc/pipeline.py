"""palm_preproc pipeline: stage orchestration and CLI.

Usage (via the root entry point):
    python run_preproc.py -c config/<project>.yaml [flags]

Stages:
    domains  domain.shp -> domain_child.shp + domain_parent.shp (snapped,
             nested, PALM-friendly sizes)
    report   domains_report.txt with per-domain grid info, decomposition
             check, nesting validation and a &nesting_parameters block
    clip     DATA_raw layers -> DATA_child/ + DATA_parent/ (vectors clipped,
             rasters resampled onto the exact PALM grid of each domain)
    merge    user child-domain layers coalesced over the clipped raw layers
             (user priority inside the original domain.shp footprint)
"""

import argparse
import shutil
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import Config, ConfigError, RASTER_LAYERS, VECTOR_LAYERS, ALL_LAYERS
from .log import debug as log_debug, get_logger, progress, setup_logging
from .state import State, config_hash
from .steps.clip import (clip_raster_task, clip_vector_task,
                         mask_raster_to_vector_task,
                         merge_raster_task)
from .steps.domains import DomainSpec, make_child_spec, make_parent_spec
from .steps.merge import coalesce_task
from .steps.boundary import clean_boundary
from .steps.report import write_domain_report
from .steps.templates import write_templates

log = get_logger()

DOMAIN_NAMES = ("child", "parent")


# ------------------------------
# 1. STAGE: DOMAINS
# ------------------------------
def stage_domains(cfg, state):
    import geopandas as gpd

    if state.is_done("domains"):
        log.info("Stage domains: already done, skipping.")
        return

    aligned = cfg["crs"]["aligned"]
    dom_out_crs = cfg["crs"]["domain_output"] or aligned

    src_path, src_kind = cfg.domain_source()
    if src_kind == "domain":
        progress("Reading the domain shapefile")
    else:
        progress("Deriving the domain from the {} layer extent",
                 cfg["user_data"]["domain_from"])
    log_debug("Domain source ({}): {}", src_kind, src_path)
    gdf = gpd.read_file(src_path)
    if gdf.crs is None:
        raise RuntimeError(f"{Path(src_path).name} has no CRS defined.")
    gdf_al = gdf.to_crs(aligned)

    # On resume, reuse the exact geometry decided on the first run (which may
    # have involved an interactive child-optimization choice) instead of
    # recomputing / re-prompting, so a resumed or non-interactive rerun can
    # never silently land on a different extent than the clipped data.
    saved_child = state.get_data("spec_child")
    saved_parent = state.get_data("spec_parent")
    if state.is_done("domains") and saved_child and saved_parent:
        child = DomainSpec.from_dict(saved_child)
        parent = DomainSpec.from_dict(saved_parent)
        log.debug("Reusing domain geometry from the resume state.")
    else:
        child = make_child_spec(
            gdf_al.total_bounds, cfg["domains"]["child"], aligned,
            parent_grid_size=cfg["domains"]["parent"]["grid_size"],
            align_to_parent=cfg["domains"]["align_child_to_parent"],
        )
        parent = make_parent_spec(child, cfg["domains"]["parent"], aligned,
                                  strict=cfg["domains"].get("strict_nesting", True))

    # If the geometry differs from a previous run (e.g. topology optimization
    # grew the child, or the confirmation prompt was answered differently),
    # any clip/merge/mask work recorded against the old extent is stale:
    # invalidate it so the enlarged region is regenerated. Without this, the
    # new ring of the child (and the parent buffered around it) would be
    # skipped and left without data.
    old_specs = {n: state.get_data(f"spec_{n}") for n in ("child", "parent")}
    new_specs = {"child": child.to_dict(), "parent": parent.to_dict()}
    if any(old_specs[n] is not None and old_specs[n] != new_specs[n]
           for n in ("child", "parent")):
        cleared = state.invalidate_prefixes(("clip:", "merge:", "mask:", "boundary:"))
        if cleared:
            log.warning(f"Domain geometry changed since the last run; "
                        f"invalidated {cleared} clip/merge/mask step(s) so "
                        f"the new extent is regenerated.")

    progress("Writing domain rectangles")
    for spec in (child, parent):
        out = cfg.output_dir / f"domain_{spec.name}.shp"
        out.parent.mkdir(parents=True, exist_ok=True)
        spec.to_gdf(dom_out_crs).to_file(out)
        log.debug(f"Written: {out}")
        state.set_data(f"spec_{spec.name}", spec.to_dict())

    # Priority footprint for the merge stage - a SINGLE footprint shared by
    # all user layers so buildings/roofs/walls/trees stay mutually consistent
    # (user data inside it, raw outside it):
    #   * explicit domain.shp  -> that polygon.
    #   * layer-extent domain  -> the ACTUAL geometry of the source layer (its
    #     real, possibly slanted/irregular footprint), NOT its bounding box:
    #     a bbox of a rotated domain is far larger than the domain, which
    #     would strip raw from areas the user never covered. Areal sources use
    #     their dissolved coverage; point/line sources fall back to the convex
    #     hull. The optimization-enlarged ring is filled with raw either way.
    if src_kind == "domain":
        footprint = gdf_al.geometry.union_all()
    else:
        footprint = gdf_al.geometry.union_all()
        if float(getattr(footprint, "area", 0.0)) <= 0:
            footprint = footprint.convex_hull
    state.set_data("priority_wkt", footprint.wkt)

    # Per-layer merge masks -> store as WKT in the aligned CRS so merge
    # workers get a picklable footprint (like priority_wkt).
    masks = cfg["merge"].get("masks", {}) or {}
    mask_wkt = {}
    for layer, mpath in masks.items():
        mg = gpd.read_file(mpath)
        if mg.crs is None:
            raise RuntimeError(f"merge mask for '{layer}' has no CRS: {mpath}")
        geom = mg.to_crs(aligned).geometry.union_all()
        if float(getattr(geom, "area", 0.0)) <= 0:
            geom = geom.convex_hull
        mask_wkt[layer] = geom.wkt
        log.debug(f"merge mask for {layer}: {mpath}")
    state.set_data("mask_wkt", mask_wkt)
    state.mark_done("domains")


def stage_report(cfg, state):
    specs = _load_specs(cfg, state)
    write_domain_report(specs["child"], specs["parent"], cfg,
                        topology=state.get_data("topology"))


def stage_boundary(cfg, state):
    if not cfg["boundary_cleanup"].get("enabled", True):
        log.debug("boundary_cleanup disabled, skipping.")
        return
    specs = _load_specs(cfg, state)
    domains = cfg["boundary_cleanup"].get("domains", DOMAIN_NAMES)
    for dom in domains:
        key = f"boundary:{dom}"
        if state.is_done(key):
            log.debug(f"{_pretty(key)}: already done, skipping.")
            continue
        msg = clean_boundary(dom, cfg, specs[dom])
        log.info(msg)
        state.mark_done(key)


def stage_templates(cfg, state):
    specs = _load_specs(cfg, state)
    write_templates(specs["child"], specs["parent"], cfg, state)


def _load_specs(cfg, state):
    specs = {}
    for name in DOMAIN_NAMES:
        d = state.get_data(f"spec_{name}")
        if d is None:
            raise RuntimeError("Domain specs missing from state; run the "
                               "'domains' stage first.")
        specs[name] = DomainSpec.from_dict(d)
    return specs


# ------------------------------
# 2. STAGE: CLIP
# ------------------------------
def _clip_output_path(cfg, dom, layer):
    """Final clip destination. Layers that will be merged afterwards go into
    an _clipped/ subdirectory; everything else lands in DATA_<dom> directly."""
    fname = cfg["raw_data"]["layers"][layer]
    if cfg.user_layer_path(layer):   # will be merged -> keep the raw clip aside
        return cfg.data_dir(dom) / "_clipped" / fname
    return cfg.data_dir(dom) / fname


def _place_aux_files(cfg):
    """Copy/create auxiliary files (surface_params.csv, import_process.log)
    into DATA_child and DATA_parent. Never overwrites existing files, so a
    log filled by the database import or an edited CSV survives reruns."""
    progress("Placing auxiliary files into the data directories")
    aux = cfg["raw_data"].get("aux_files", {}) or {}
    for dom in DOMAIN_NAMES:
        ddir = cfg.data_dir(dom)
        ddir.mkdir(parents=True, exist_ok=True)
        for fname, mode in aux.items():
            dst = ddir / fname
            if dst.exists() and not cfg["project"].get("overwrite"):
                log.debug(f"[clip] {dom}/{fname}: exists, kept.")
                continue
            if mode == "copy":
                src = cfg["raw_data"]["dir"] / fname
                if not src.exists():
                    log.warning(f"[clip] {dom}/{fname}: source missing in "
                                f"{cfg['raw_data']['dir']}; skipped.")
                    continue
                shutil.copy2(src, dst)
                log.debug(f"{dom}/{fname}: copied from raw_data")
            elif mode in ("touch", "create_empty"):
                dst.write_text("")   # touch() would keep existing content
                log.debug(f"{dom}/{fname}: created empty")
            else:
                log.warning(f"[clip] {dom}/{fname}: unknown mode '{mode}'; skipped.")


def stage_clip(cfg, state):
    _place_aux_files(cfg)
    specs = _load_specs(cfg, state)
    progress("Clipping vector layers to the domains")
    aligned = cfg["crs"]["aligned"]
    snap = cfg["clip"]["snap_rasters_to_grid"]
    res_cfg = cfg["clip"]["resampling"]

    tasks = []   # (key, fn, args)
    for dom in DOMAIN_NAMES:
        spec = specs[dom]
        rect_wkt = spec.rectangle().wkt
        for layer in ALL_LAYERS:
            key = f"clip:{dom}:{layer}"
            if state.is_done(key):
                log.debug(f"{_pretty(key)}: already done, skipping.")
                continue
            src = cfg.raw_layer_path(layer)
            if not src.exists():
                log.warning(f"{_pretty(key)}: raw layer missing ({src}); skipping.")
                continue
            out = _clip_output_path(cfg, dom, layer)
            if layer in RASTER_LAYERS:
                resampling = res_cfg.get(layer, res_cfg.get("default", "nearest"))
                force_nodata = (cfg["clip"].get("buildings_nodata")
                                if layer == "buildings" else None)
                tasks.append((key, clip_raster_task,
                              (str(src), str(out), spec.to_dict(), resampling,
                               snap, force_nodata)))
            else:
                tasks.append((key, clip_vector_task,
                              (str(src), str(out), rect_wkt, aligned)))

    _run_parallel("clip", tasks, cfg["clip"]["workers"], state)
    _mask_buildings(cfg, state, after_merge=False)


def _mask_buildings(cfg, state, after_merge):
    """Clip buildings.tif to the FINAL roofs layer of each domain (remark:
    building heights must exist only under roof polygons). Runs at the end
    of `clip` for domains whose roofs come straight from raw_data, and at
    the end of `merge` for domains whose roofs are user-merged."""
    mask_layer = cfg["clip"].get("buildings_mask_layer")
    if not mask_layer:
        return
    # The mask needs the FINAL buildings and FINAL mask layer. Defer it to
    # after the merge stage if either the mask layer (roofs) or buildings
    # itself is user-merged; otherwise it can run at the end of clip.
    deferred = (mask_layer in cfg["user_data"]["layers"]
                or "buildings" in cfg["user_data"]["layers"])
    if after_merge != deferred:
        return
    progress("Clipping the buildings raster to the roofs layer")
    nodata = cfg["clip"].get("buildings_nodata", -9999.0)
    tasks = []
    for dom in DOMAIN_NAMES:
        key = f"mask:{dom}:buildings"
        if state.is_done(key):
            log.debug(f"{_pretty(key)}: already done, skipping.")
            continue
        raster = cfg.data_dir(dom) / cfg["raw_data"]["layers"]["buildings"]
        vector = cfg.data_dir(dom) / cfg["raw_data"]["layers"][mask_layer]
        if not raster.exists() or not vector.exists():
            log.warning(f"{_pretty(key)}: buildings or {mask_layer} layer "
                        f"missing; mask skipped.")
            continue
        tasks.append((key, mask_raster_to_vector_task,
                      (str(raster), str(vector), nodata)))
    _run_parallel("mask", tasks, cfg["clip"]["workers"], state)


# ------------------------------
# 3. STAGE: MERGE
# ------------------------------
def stage_merge(cfg, state):
    user_layers = cfg["user_data"]["layers"]
    if not user_layers:
        log.debug("No user layers configured, nothing to merge")
        return

    specs = _load_specs(cfg, state)
    priority_wkt = state.get_data("priority_wkt")
    if priority_wkt is None:
        raise RuntimeError("Priority footprint missing from state; run the "
                           "'domains' stage first.")
    aligned = cfg["crs"]["aligned"]
    mode = cfg["merge"]["mode"]
    predicate = cfg["merge"]["predicate"]
    priority_source = cfg["merge"].get("priority_source", "domain")
    fix_geometry = cfg["merge"].get("fix_geometry", True)
    int_columns = cfg["merge"].get("int_columns", []) or []
    reassign_map = cfg["merge"].get("reassign_id", {}) or {}
    mark_user = cfg["merge"].get("mark_user_column")
    raw_fill = cfg["merge"].get("raw_fill", {}) or {}
    mask_wkt = state.get_data("mask_wkt", {}) or {}

    res_cfg = cfg["clip"]["resampling"]
    tasks = []
    for dom in DOMAIN_NAMES:
        spec = specs[dom]
        rect_wkt = spec.rectangle().wkt
        for layer, user_src in user_layers.items():
            key = f"merge:{dom}:{layer}"
            if state.is_done(key):
                log.debug(f"{_pretty(key)}: already done, skipping.")
                continue
            clipped = cfg.data_dir(dom) / "_clipped" / cfg["raw_data"]["layers"][layer]
            out = cfg.data_dir(dom) / cfg["raw_data"]["layers"][layer]

            if layer in RASTER_LAYERS:
                if not clipped.exists():
                    log.warning(f"{_pretty(key)}: clipped raw raster missing "
                                f"({clipped.name}); skipping raster merge.")
                    continue
                resampling = res_cfg.get(layer, res_cfg.get("default", "nearest"))
                force_nodata = (cfg["clip"].get("buildings_nodata")
                                if layer == "buildings" else None)
                tasks.append((key, merge_raster_task,
                              (str(user_src), str(clipped), str(out),
                               spec.to_dict(), priority_wkt, resampling,
                               force_nodata)))
            else:
                if not clipped.exists():
                    log.warning(f"{_pretty(key)}: clipped raw layer missing "
                                f"({clipped.name}); merging user layer alone.")
                    clipped = None
                tasks.append((key, coalesce_task,
                              (str(user_src), str(clipped) if clipped else None,
                               str(out), priority_wkt, rect_wkt, aligned,
                               mode, predicate, priority_source,
                               fix_geometry, tuple(int_columns),
                               reassign_map.get(layer), mark_user,
                               raw_fill.get(layer, "footprint"),
                               mask_wkt.get(layer))))

    _run_parallel("merge", tasks, cfg["merge"]["workers"], state)
    _mask_buildings(cfg, state, after_merge=True)


# ------------------------------
# 4. PARALLEL RUNNER
# ------------------------------
def _pretty(key):
    parts = key.split(":")
    return f"[{parts[0]}] " + "/".join(parts[1:])


def _run_parallel(stage, tasks, workers, state):
    if not tasks:
        log.debug(f"Stage {stage}: nothing to do")
        return
    log.debug(f"Stage {stage}: {len(tasks)} task(s), {workers} worker(s)")
    failed = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fn, *args): key for key, fn, args in tasks}
        for fut in as_completed(futures):
            key = futures[fut]
            try:
                msg = fut.result()
            except Exception as exc:
                log.error(f"{_pretty(key)}: FAILED -- {exc}")
                failed += 1
                continue
            if msg.startswith("ERROR"):
                log.error(f"{_pretty(key)}: {msg}")
                failed += 1
            elif msg.startswith("WARNING"):
                log.warning(f"{_pretty(key)}: {msg}")
                state.mark_done(key)   # empty result is a valid outcome
            else:
                for line in msg.splitlines():
                    log.debug(f"{_pretty(key)}: {line}")
                state.mark_done(key)
    if failed:
        raise RuntimeError(f"Stage {stage}: {failed} task(s) failed.")
    log.debug(f"Stage {stage}: {len(tasks)} task(s) done")


# ------------------------------
# 5. DRY RUN
# ------------------------------
def print_plan(cfg, state, stages):
    """Validate the config and print what would be done, writing nothing."""
    log.info("DRY RUN - no files will be written.")
    log.info(f"Domain input: {cfg['user_data']['domain']}")
    for name in DOMAIN_NAMES:
        dc = cfg["domains"][name]
        log.info(f"  {name}: grid {dc['grid_size']:g} m, buffer {dc.get('buffer', 0):g} m, "
                 f"optimize_topology={dc.get('optimize_topology', False)}")
    for stage in stages:
        if stage == "domains":
            status = "done" if state.is_done("domains") else "pending"
            log.info(f"[domains] {status} -> domain_child.shp, domain_parent.shp")
        elif stage == "boundary":
            if not cfg["boundary_cleanup"].get("enabled", True):
                log.info("[boundary] disabled")
                continue
            for dom in cfg["boundary_cleanup"].get("domains", DOMAIN_NAMES):
                key = f"boundary:{dom}"
                status = "done" if state.is_done(key) else "pending"
                log.info(f"[{key}] {status}")
        elif stage == "templates":
            case = cfg["templates"]["case"] or cfg["project"]["name"]
            names = ([f"{case}_p3d", f"{case}_p3d_N02", f"{case}_p3dr",
                      f"{case}_p3dr_N02", f"pgem_{case}.yaml",
                      f"pgem_{case}_N02.yaml", f"pmeteo_{case}.yaml",
                      f"pmeteo_{case}_N02.yaml"]
                     if cfg["templates"].get("nested", True) else
                     [f"{case}_p3d", f"{case}_p3dr",
                      f"pgem_{case}.yaml", f"pmeteo_{case}.yaml"])
            for n in names:
                dst = cfg.output_dir / n
                status = "exists, kept" if dst.exists() else "pending"
                log.info(f"[templates:{n}] {status} -> {dst}")
        elif stage == "report":
            log.info(f"[report] always regenerated -> "
                     f"{cfg['report']['file'] or cfg.output_dir / 'domains_report.txt'}")
        elif stage == "clip":
            for dom in DOMAIN_NAMES:
                for fname, mode in (cfg["raw_data"].get("aux_files", {}) or {}).items():
                    dst = cfg.data_dir(dom) / fname
                    status = "exists, kept" if dst.exists() else f"pending ({mode})"
                    log.info(f"[aux:{dom}:{fname}] {status} -> {dst}")
                for layer in ALL_LAYERS:
                    key = f"clip:{dom}:{layer}"
                    src = cfg.raw_layer_path(layer)
                    status = ("done" if state.is_done(key)
                              else "MISSING SOURCE" if not src.exists() else "pending")
                    log.info(f"[{key}] {status} -> {_clip_output_path(cfg, dom, layer)}")
        elif stage == "merge":
            if not cfg["user_data"]["layers"]:
                log.info("[merge] no user layers -> nothing to do")
                continue
            for dom in DOMAIN_NAMES:
                for layer in cfg["user_data"]["layers"]:
                    key = f"merge:{dom}:{layer}"
                    status = "done" if state.is_done(key) else "pending"
                    log.info(f"[{key}] {status} -> "
                             f"{cfg.data_dir(dom) / cfg['raw_data']['layers'][layer]}")


# ------------------------------
# 6. MAIN
# ------------------------------
STAGE_LABELS = {
    "domains": "Computing domain geometry",
    "clip": "Preparing domain data",
    "merge": "Merging user data into the domain data",
    "boundary": "Cleaning up domain-boundary buildings",
    "templates": "Preparing run configuration",
    "report": "Writing case report",
}

STAGE_FUNCS = {"domains": stage_domains, "report": stage_report,
               "templates": stage_templates, "boundary": stage_boundary,
               "clip": stage_clip, "merge": stage_merge}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="palm_preproc",
                                 description="PALM static-driver input preprocessing.")
    ap.add_argument("-c", "--config", required=True, help="YAML config file")
    ap.add_argument("--force", action="store_true",
                    help="ignore previous resume state and rerun everything")
    ap.add_argument("--stages", nargs="+", default=None,
                    help="subset of stages to run (default: from config)")
    ap.add_argument("-n", "--dry-run", action="store_true",
                    help="validate config and print plan, write nothing")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="DEBUG-level logging")
    ap.add_argument("-q", "--quiet", action="store_true",
                    help="WARNING-level logging only")
    ap.add_argument("--log-datetime", action="store_true",
                    help="full date in timestamps (time-only is the default)")
    args = ap.parse_args(argv)

    verbosity = "debug" if args.verbose else ("warning" if args.quiet else "info")
    setup_logging(verbosity, args.log_datetime)

    # Version + interactivity up front: a batch job's log should say which
    # code produced it, and that prompts were auto-answered (otherwise the
    # same config behaves "differently" on a login node, which looks like a
    # malfunction).
    from . import __version__
    log.info(f"palm_preproc {__version__}")
    if not sys.stdin.isatty():
        log.info("non-interactive session: prompts auto-accept the "
                 "recommended option (child sizing confirmation, "
                 "processor topology choice)")

    try:
        cfg = Config(args.config)
    except (ConfigError, FileNotFoundError) as exc:
        log.error(f"Config error: {exc}")
        return 1

    overwrite_all = bool(cfg["project"].get("overwrite"))
    if overwrite_all:
        log.warning("project.overwrite: true - regenerating ALL outputs "
                    "(resume state ignored; templates and aux files replaced).")
    state = State(cfg["project"]["state_file"],
                  config_hash(cfg.as_plain_dict()),
                  force=args.force or overwrite_all)

    stages = args.stages or cfg["stages"]
    progress("Reading configuration")
    log_debug("Project: {}", cfg["project"]["name"])
    log_debug("Stages: {}", " -> ".join(stages))
    log_debug("Output dir: {}", cfg.output_dir)
    log_debug("User layers: {}", ", ".join(sorted(cfg["user_data"]["layers"]))
              or "none (clipped raw data only)")

    if args.dry_run:
        print_plan(cfg, state, stages)
        return 0

    for stage in stages:
        fn = STAGE_FUNCS.get(stage)
        if fn is None:
            log.error(f"Unknown stage: {stage}")
            return 1
        progress(STAGE_LABELS.get(stage, stage))
        try:
            fn(cfg, state)
        except Exception as exc:
            log.error(f"Stage {stage} failed: {exc}")
            return 1

    progress("palm_preproc finished OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
