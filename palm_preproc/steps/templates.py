"""Run-configuration template instantiation for palm_preproc.

Fills the pgem (static-driver generator), pmeteo (palm_meteo dynamic
driver) and PALM _p3d namelist templates with the computed domain values and
writes them into the output directory:

  nested (default):
    <case>_p3d, <case>_p3d_N02, <case>_p3dr, <case>_p3dr_N02,
    pgem_<case>.yaml, pgem_<case>_N02.yaml,
    pmeteo_<case>.yaml, pmeteo_<case>_N02.yaml
  non-nested:
    <case>_p3d, <case>_p3dr, pgem_<case>.yaml, pmeteo_<case>.yaml

_p3dr files are derived from the filled _p3d files by swapping the
initialization: the active `initializing_actions` line is commented out and
the commented `read_restart_data` line is activated (reference_state,
dt_run_control_1d and dt_pr_1d are already present in the templates).

Values not derivable from the domain specs (origin_time, simulation length,
nz, wrf_date) come from `templates.values` in the config; when missing,
the placeholder is left in the file and a warning lists what remains for
manual filling. Existing output files are never overwritten unless
templates.overwrite is true.
"""

import math
import os
import re
from pathlib import Path

from ..log import get_logger, progress
from .domains import snap_up, valid_pairs
from .topology import (best_per_node_group, choose_topology,
                       dedupe_by_cores, filter_by_nodes,
                       nested_configurations,
                       recommend_nested, recommend_single,
                       single_configurations)

log = get_logger()

BUILTIN_DIR = Path(__file__).resolve().parent.parent.parent / "defaults" / "templates"

# (per-domain rules live in steps/domains.py; enumeration, joint nested
#  constraint and selection in steps/topology.py - rules by Hynek Reznicek)

_PLACEHOLDER_RE = re.compile(r"<[A-Za-z0-9_.\- ]*>")


# ------------------------------
# 1. HELPERS
# ------------------------------
def _sub_keyed(text, key, value, sep=r"=", suffix=""):
    """Replace `key <sep> <>` (+optional literal suffix like '.0') on any
    line, keeping everything around it."""
    pattern = re.compile(rf"({re.escape(key)}\s*{sep}\s*)<>{re.escape(suffix)}")
    return pattern.sub(lambda m: f"{m.group(1)}{value}", text)


def _sub_token(text, token, value):
    return text.replace(token, str(value))


def _sub_line_positional(text, line_marker, values, suffix=""):
    """On lines containing `line_marker`, replace successive `<>`+suffix
    occurrences with the given values (left to right)."""
    out = []
    for line in text.splitlines(keepends=True):
        if line_marker in line:
            for v in values:
                line = line.replace(f"<>{suffix}", str(v), 1)
        out.append(line)
    return "".join(out)


def _fmt_g(v):
    return f"{float(v):g}"


# ------------------------------
# 2. CONTEXT
# ------------------------------
def _max_building_height(cfg, domain_name):
    """Max valid value in the domain's FINAL buildings raster, or None if
    it can't be determined (missing file, all-nodata)."""
    import numpy as np

    path = cfg.data_dir(domain_name) / cfg["raw_data"]["layers"]["buildings"]
    if not path.exists():
        return None
    from .clip import _import_rasterio
    rasterio, _ = _import_rasterio()
    with rasterio.open(path) as ds:
        arr = ds.read(1).astype("float64")
        nodata = ds.nodata
        if nodata is not None:
            arr = arr[arr != nodata]
        arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else None


def _mean_abs_rooftop_height(cfg, domain_name):
    """Mean absolute rooftop height over the domain's building cells, in
    metres, referenced to the domain's lowest terrain point (the datum PALM's
    static driver uses: min(zt) -> z = 0). Or None if it can't be determined.

    For each building cell: abs_roof = terrain + building_height - min(terrain
    over the whole domain). The average is taken over building cells only, so
    a hill with no buildings on it raises min(terrain) into the datum but does
    NOT inflate the target; buildings sitting on higher ground correctly count
    at their true absolute rooftop height. This matches the roughness-sublayer
    reasoning (dz*nz ~ 3-5 x mean building height) in the static driver's
    absolute-height frame.

    Requires the DEM and buildings rasters to be on the same grid (they are
    when clip.snap_rasters_to_grid is on, which snaps both to each domain's
    exact PALM grid). Returns None (with a warning) if they are not, so the
    caller can fall back to the relative-height estimate.
    """
    import numpy as np

    ddir = cfg.data_dir(domain_name)
    dem_path = ddir / cfg["raw_data"]["layers"]["dem"]
    bld_path = ddir / cfg["raw_data"]["layers"]["buildings"]
    if not dem_path.exists() or not bld_path.exists():
        return None

    from .clip import _import_rasterio
    rasterio, _ = _import_rasterio()
    with rasterio.open(dem_path) as dds:
        dem = dds.read(1).astype("float64")
        dem_nodata = dds.nodata
    with rasterio.open(bld_path) as bds:
        bld = bds.read(1).astype("float64")
        bld_nodata = bds.nodata

    if dem.shape != bld.shape:
        log.warning(f"{domain_name}: DEM {dem.shape} and buildings "
                    f"{bld.shape} rasters are not grid-aligned; cannot use "
                    f"absolute rooftop height (is clip.snap_rasters_to_grid "
                    f"enabled?).")
        return None

    # Domain datum: lowest VALID terrain point (nodata / non-finite excluded).
    terr_valid = np.isfinite(dem)
    if dem_nodata is not None:
        terr_valid &= (dem != dem_nodata)
    if not terr_valid.any():
        return None
    floor = float(dem[terr_valid].min())

    # Building cells: buildings raster valid and > 0, AND terrain valid there
    # (an absolute rooftop needs a terrain value under it).
    bld_valid = np.isfinite(bld) & (bld > 0.0)
    if bld_nodata is not None:
        bld_valid &= (bld != bld_nodata)
    mask = bld_valid & terr_valid
    if not mask.any():
        return None

    abs_roof = dem[mask] + bld[mask] - floor
    return float(abs_roof.mean())


def _auto_nz(cfg, spec):
    """Default vertical extent (grid points), used when nz_<domain> is not
    set explicitly in templates.values:
      * child:  dz*nz = child_nz_height_factor x mean absolute rooftop height
                (terrain + building, referenced to the domain's lowest terrain
                point) over the FINAL buildings raster -- the roughness-
                sublayer target in the static driver's absolute-height frame.
                Falls back to the relative max building height if the DEM and
                buildings rasters are not grid-aligned.
      * parent: dz*nz ~= parent_target_height_m.
    Both the factor/target and the resulting nz remain fully overridable.

    PALM's multigrid pressure solver needs nx, ny AND nz each divisible by
    2^levels for full coarsening depth (subdomain grid points along x, y, z
    must be a multiple of 2 per level; see docs.palm-model.com, Pressure
    solver usage). Unlike nx/ny, nz is NOT subject to the npex/npey
    processor-topology rules -- PALM's standard decomposition splits only
    x/y among MPI ranks, so subdomain nz equals the global nz directly and
    only needs plain divisibility, using the domain's own min_power (same
    stride nx/ny are snapped to).
    """
    tv = cfg["templates"]["values"]
    dz = spec.grid_size
    min_power = cfg["domains"][spec.name].get("min_power", 3)
    stride = 1 << min_power
    if spec.name == "child":
        factor = float(tv.get("child_nz_height_factor", 3.0))
        h = _mean_abs_rooftop_height(cfg, spec.name)
        basis = "mean absolute rooftop height"
        if h is None:
            # Grids not aligned, or no DEM: fall back to the relative
            # (above-terrain) building height so a driver is still produced.
            # Note this can undersize nz where buildings sit well above the
            # domain's lowest terrain.
            h = _max_building_height(cfg, spec.name)
            basis = "max relative building height (DEM unavailable/unaligned)"
        if h is None:
            log.warning(f"{spec.name}: could not determine building height "
                        f"for nz auto-calculation (buildings raster missing "
                        f"or empty); nz left as a placeholder.")
            return None
        nz_raw = factor * h / dz
        nz = snap_up(nz_raw, stride)
        log.info(f"{spec.name}: nz auto-calculated = {nz} (dz*nz = "
                 f"{factor:g} x {basis} {h:.1f} m -> "
                 f"{nz_raw:.1f} pts snapped up to a multiple of {stride} "
                 f"for multigrid)")
        return nz
    else:
        target = float(tv.get("parent_target_height_m", 2000.0))
        nz_raw = target / dz
        nz = snap_up(nz_raw, stride)
        log.info(f"{spec.name}: nz auto-calculated = {nz} (dz*nz ~= "
                 f"{target:g} m -> {nz_raw:.1f} pts snapped up to a "
                 f"multiple of {stride} for multigrid)")
        return nz


# ------------------------------
# (helper) seasonal water / soil temperatures from origin_time month
# ------------------------------
_SEASONAL_STATS = ("min", "ave", "max")

# one-shot flags for warnings that would otherwise repeat per domain
_WARNED = {}


def _cfg_get(cfg, key, default=None):
    """Read a top-level section from either a Config object (which supports
    only __getitem__) or a plain dict."""
    try:
        return cfg[key]
    except (KeyError, TypeError):
        return default


def _merge_seasonal(base, over):
    """Recursive dict merge: `over` wins, nested dicts merge."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_seasonal(out[k], v)
        else:
            out[k] = v
    return out


def _load_seasonal(cfg):
    """The seasonal tables live in seasonal.yaml NEXT TO the template
    skeletons (defaults/templates/, or the project's templates.dir), since
    they parameterize the templates and should travel with them. Keys under
    a `seasonal:` section in the merged config deep-merge OVER the file, so
    a project can e.g. flip `statistic: max` without editing the shared
    file."""
    tcfg = _cfg_get(cfg, "templates") or {}
    tdir = Path(tcfg["dir"]) if tcfg.get("dir") else BUILTIN_DIR
    file_cfg = {}
    path = tdir / "seasonal.yaml"
    if path.exists():
        import yaml
        try:
            file_cfg = yaml.safe_load(path.read_text()) or {}
        except Exception as e:
            log.warning(f"seasonal: cannot parse {path}: {e}; "
                        f"seasonal temperatures not applied.")
            file_cfg = {}
    return _merge_seasonal(file_cfg, _cfg_get(cfg, "seasonal") or {})


def _seasonal_key(scfg, canonical, deprecated):
    """Read a seasonal table under its canonical name, accepting the old
    name with a deprecation warning."""
    if scfg.get(canonical) is not None:
        return scfg.get(canonical)
    if scfg.get(deprecated) is not None:
        log.warning(f"seasonal: key '{deprecated}' is deprecated; "
                    f"rename it to '{canonical}' (canonical PALM/PALM-GeM "
                    f"name). Values are used unchanged.")
        return scfg.get(deprecated)
    return None


def _seasonal_values(cfg, origin_time):
    """Month-dependent temperatures from seasonal.yaml (see _load_seasonal),
    keyed on the month of templates.values.origin_time. Returns a dict:
      water:     {pgem water type id (int): temperature [K] (float)}
      soil:      list of 8 layer temperatures [K], or None
      deep_soil: temperature [K], or None
      p3d:       {p3d namelist parameter name: temperature [K]}
    All tables are entered in KELVIN (12 monthly values, Jan..Dec) and
    substituted verbatim - no unit conversion. `statistic` picks the min/ave/max column. Anything
    not configured (or enabled: false, or a missing/unparseable
    origin_time) leaves the template's own values untouched.
    """
    scfg = _load_seasonal(cfg)
    empty = {"water": {}, "soil": None, "deep_soil": None,
             "soil_moisture": None, "p3d": {}}
    if not scfg or not scfg.get("enabled", True) or not origin_time:
        return empty
    m = re.match(r"\s*\d{4}-(\d{2})-\d{2}", str(origin_time))
    if not m:
        log.warning(f"seasonal: cannot read a month from origin_time "
                    f"'{origin_time}'; seasonal temperatures not applied.")
        return empty
    month_idx = int(m.group(1)) - 1          # 0-based Jan..Dec
    stat = str(scfg.get("statistic", "ave")).lower()
    if stat not in _SEASONAL_STATS:
        log.warning(f"seasonal.statistic '{stat}' not one of "
                    f"{_SEASONAL_STATS}; using 'ave'.")
        stat = "ave"

    def month_value(table, what):
        """One Kelvin value for this month from a 12-entry list or a
        {min/ave/max: 12-entry list} mapping. When the requested statistic
        column is absent, falls back to 'ave' (so e.g. a max-statistic run
        still uses tables that only provide averages)."""
        if isinstance(table, dict):
            table = table.get(stat, table.get("ave"))
        if table is None:
            return None
        if not isinstance(table, (list, tuple)) or len(table) != 12:
            log.warning(f"seasonal: {what} needs 12 monthly values "
                        f"(Jan..Dec); got {table!r}. Not applied.")
            return None
        return float(table[month_idx])

    out = {"water": {}, "soil": None, "deep_soil": None,
           "soil_moisture": None, "p3d": {}}
    old_keys = [k for k in ("water_temperature_c", "soil_temperature_c",
                            "deep_soil_temperature_c", "p3d_temperature_c")
                if k in scfg]
    if old_keys:
        log.warning(f"seasonal: found old Celsius-schema key(s) {old_keys}; "
                    f"tables are now in KELVIN with _k suffixes "
                    f"(e.g. water_temperature_k) - these entries are ignored.")
    for type_id, table in (_seasonal_key(scfg, "water_pars_temp",
                                     "water_temperature_k") or {}).items():
        k = month_value(table, f"water_pars_temp[{type_id}]")
        if k is not None:
            out["water"][int(type_id)] = k
    ph = set(int(i) for i in (scfg.get("placeholder_water_types") or []))
    used_ph = sorted(ph & set(out["water"]))
    if used_ph and not _WARNED.get("placeholder_water"):
        _WARNED["placeholder_water"] = True   # once per run, not per domain
        log.warning(f"seasonal: water type(s) {used_ph} still use placeholder "
                    f"year-round tables (TODO) - replace them with measured "
                    f"monthly data in seasonal.yaml and remove the ids from "
                    f"placeholder_water_types.")

    soil = _seasonal_key(scfg, "soil_temperature", "soil_temperature_k")
    if soil is not None:
        c = month_value(soil, "soil_temperature")
        layers = None
        if c is not None:                     # 12 scalars: uniform layers
            layers = [c] * 8
        elif (isinstance(soil, (list, tuple)) and len(soil) == 12
              and all(isinstance(v, (list, tuple)) for v in soil)):
            layer_row = soil[month_idx]       # 12 rows of 8 layer values
            if len(layer_row) == 8:
                layers = [float(v) for v in layer_row]
            else:
                log.warning(f"seasonal: soil_temperature month rows need "
                            f"8 layer values; got {len(layer_row)}.")
        if layers:
            out["soil"] = list(layers)

    deep = _seasonal_key(scfg, "deep_soil_temperature",
                         "deep_soil_temperature_k")
    if deep is not None:
        if isinstance(deep, (int, float)):    # single annual value
            out["deep_soil"] = float(deep)
        else:
            c = month_value(deep, "deep_soil_temperature")
            if c is not None:
                out["deep_soil"] = c

    # soil moisture: DIMENSIONLESS volumetric fraction - no C -> K
    # conversion. Same table forms as soil_temperature_k: 12 monthly
    # scalars (uniform over the 8 layers), a {min/ave/max: ...} mapping,
    # or 12 rows of 8 per-layer values.
    moist = scfg.get("soil_moisture")
    if moist is not None:
        c = month_value(moist, "soil_moisture")
        layers = None
        if c is not None:
            layers = [c] * 8
        elif (isinstance(moist, (list, tuple)) and len(moist) == 12
              and all(isinstance(v, (list, tuple)) for v in moist)):
            layer_row = moist[month_idx]
            if len(layer_row) == 8:
                layers = [float(v) for v in layer_row]
            else:
                log.warning(f"seasonal: soil_moisture month rows need "
                            f"8 layer values; got {len(layer_row)}.")
        if layers:
            out["soil_moisture"] = layers

    # other scalar p3d values by namelist parameter name, substituted
    # verbatim (K, or the parameter's own unit - deltas like
    # spinup_pt_amplitude are safe here since nothing is converted)
    for param, table in (_seasonal_key(scfg, "p3d_parameters",
                                   "p3d_temperature_k") or {}).items():
        k = month_value(table, f"p3d_parameters[{param}]")
        if k is not None:
            out["p3d"][str(param)] = k
    return out


def _apply_water_temps(text, temps):
    """Rewrite `<id>: <value>` lines inside the pgem `water_pars_temp:` block
    for the type ids present in `temps` (id -> K). Lines for other ids, and
    everything outside the block, are untouched; trailing comments are kept.
    """
    if not temps:
        return text
    lines = text.splitlines(keepends=True)
    in_block = False
    block_indent = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"water_pars_temp\s*:", stripped):
            in_block = True
            block_indent = len(line) - len(line.lstrip())
            continue
        if in_block:
            if stripped and (len(line) - len(line.lstrip())) <= block_indent:
                in_block = False              # dedent: block over
                continue
            m = re.match(r"^(\s*)(\d+)(\s*:\s*)([0-9.]+)(.*)$",
                         line.rstrip("\n"))
            if m and int(m.group(2)) in temps:
                k = temps[int(m.group(2))]
                lines[i] = (f"{m.group(1)}{m.group(2)}{m.group(3)}"
                            f"{k:.2f}{m.group(5)}\n")
    return "".join(lines)


def _parse_hours(value):
    """Simulation length in hours. Accepts a bare number (hours) or a
    duration string with an explicit unit: '29 h', '1 d', '90 min', '3600 s'
    (palm_meteo's duration style). Returns float hours, or None."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    m = re.match(r"\s*([0-9.]+)\s*(h|hr|hours?|d|days?|m|min|minutes?|s|secs?|seconds?)\s*$",
                 str(value), re.I)
    if not m:
        log.warning(f"templates.values.length: cannot parse duration "
                    f"{value!r}; expected a number of hours or e.g. '29 h', "
                    f"'1 d'. Leaving unset.")
        return None
    n, unit = float(m.group(1)), m.group(2).lower()[0]
    return {"h": n, "d": n * 24.0, "m": n / 60.0, "s": n / 3600.0}[unit]


def build_context(spec, cfg, other=None):
    """Assemble the substitution context for one domain.

    `spec` is the DomainSpec the file describes; `other` is the counterpart
    domain (needed by the nested parent namelist, which also carries the
    child core count and offsets)."""
    tv = cfg["templates"]["values"]
    name = spec.name
    topo_opt = cfg["domains"]["parent"].get("topology_opt", True)

    npex = tv.get(f"npex_{name}")
    npey = tv.get(f"npey_{name}")
    if not (npex and npey):
        log.warning(f"{name}: npex/npey not resolved, left as placeholders")
        npex = npey = None

    nz = tv.get(f"nz_{name}") or tv.get("nz")
    if not nz:
        nz = _auto_nz(cfg, spec)
    ctx = {
        "case": cfg["templates"]["case"] or cfg["project"]["name"],
        "spec": spec,
        "other": other,
        "nz": nz,
        "npex": npex,
        "npey": npey,
        "origin_time": tv.get("origin_time"),
        # "length" is the current key; "simulation_hours" accepted for
        # backwards compatibility with older configs
        "hours": _parse_hours(tv.get("length")
                              if tv.get("length") is not None
                              else tv.get("simulation_hours")),
        "wrf_date": tv.get("wrf_date"),
        "hpc_user": tv.get("hpc_user"),
        "wrf_dir": tv.get("wrf_dir"),
        "seasonal": _seasonal_values(cfg, tv.get("origin_time")),
    }
    return ctx


def _resolve_topology(child, parent, cfg, nested, state=None):
    """Fill templates.values npex/npey via Hynek Reznicek's topology rules,
    interactively when configured and possible. Pinned values win. The
    resolved topology is persisted to the pipeline state so the report can
    document the actual choice."""
    tv = cfg["templates"]["values"]
    raw_mode = cfg["templates"].get("topology_select", True)
    mode = "ask" if raw_mode in (True, "ask", "true") else "auto"
    node_cpus = tv.get("node_cpus")
    topo_opt = cfg["domains"]["parent"].get("topology_opt", True)
    topo_opt_child = cfg["domains"]["child"].get("topology_opt", topo_opt)

    progress("Selecting processor topology")
    if nested:
        if all(tv.get(k) for k in ("npex_parent", "npey_parent",
                                   "npex_child", "npey_child")):
            _persist_topology(tv, state)
            return
        configs = nested_configurations(parent, child, topo_opt, topo_opt_child)
        configs = filter_by_nodes(configs, node_cpus, tv.get("min_nodes"),
                                  tv.get("max_nodes"), nested=True)
        configs = dedupe_by_cores(configs, nested=True)
        if not cfg["templates"].get("list_all_topologies", False):
            configs = best_per_node_group(configs, node_cpus, nested=True)
        rec = recommend_nested(configs, tv.get("npes_parent"),
                               tv.get("npes_child"), node_cpus)
        idx = choose_topology(configs, rec, mode, node_cpus, nested=True)
        if idx is None:
            return
        chosen = configs[idx]
        tv["npex_parent"], tv["npey_parent"] = (chosen["parent"]["npex"],
                                                chosen["parent"]["npey"])
        tv["npex_child"], tv["npey_child"] = (chosen["child"]["npex"],
                                              chosen["child"]["npey"])
        log.info(f"topology: parent npex={tv['npex_parent']} "
                 f"npey={tv['npey_parent']} ({chosen['parent']['cores']}), "
                 f"N02 npex={tv['npex_child']} npey={tv['npey_child']} "
                 f"({chosen['child']['cores']}), sum_cpu={chosen['sum']}")
    else:
        name = cfg["templates"].get("single_domain", "parent")
        spec = parent if name == "parent" else child
        if tv.get(f"npex_{name}") and tv.get(f"npey_{name}"):
            _persist_topology(tv, state)
            return
        opt = topo_opt if name == "parent" else topo_opt_child
        configs = single_configurations(spec, opt)
        configs = filter_by_nodes(configs, node_cpus, tv.get("min_nodes"),
                                  tv.get("max_nodes"), nested=False)
        configs = dedupe_by_cores(configs, nested=False)
        if not cfg["templates"].get("list_all_topologies", False):
            configs = best_per_node_group(configs, node_cpus, nested=False)
        rec = recommend_single(configs, tv.get(f"npes_{name}"), node_cpus)
        idx = choose_topology(configs, rec, mode, node_cpus, nested=False)
        if idx is None:
            return
        chosen = configs[idx]
        tv[f"npex_{name}"], tv[f"npey_{name}"] = chosen["npex"], chosen["npey"]
        log.info(f"topology: {name} npex={chosen['npex']} "
                 f"npey={chosen['npey']} ({chosen['cores']} cores)")
    _persist_topology(tv, state)


def _persist_topology(tv, state):
    if state is None:
        return
    topo = {k: tv.get(k) for k in ("npex_parent", "npey_parent",
                                   "npex_child", "npey_child") if tv.get(k)}
    if topo:
        state.set_data("topology", topo)


# ------------------------------
# 3. FILLERS
# ------------------------------
def fill_p3d(text, ctx, child_ctx=None):
    s, o = ctx["spec"], ctx["other"]
    t = text

    # header / descriptive tokens
    t = _sub_token(t, "<case_name>", ctx["case"])
    t = _sub_token(t, "<domain_name>", ctx["case"])
    t = _sub_token(t, "<run_id>", ctx["case"])
    t = _sub_token(t, "<x-1>", s.nx)
    t = _sub_token(t, "<y-1>", s.ny)
    t = _sub_token(t, "<x.dx>", _fmt_g(s.width_m))
    t = _sub_token(t, "<y.dy>", _fmt_g(s.height_m))
    t = _sub_token(t, "<x>", s.width_pts)
    t = _sub_token(t, "<y>", s.height_pts)
    t = _sub_token(t, "<domain_resolution>", _fmt_g(s.grid_size))
    if ctx["nz"]:
        t = _sub_token(t, "<z.dz>", _fmt_g(ctx["nz"] * s.grid_size))
        t = _sub_token(t, "<z>", ctx["nz"])

    # nesting_parameters block (nested parent template only)
    if child_ctx is not None:
        c = child_ctx["spec"]
        if ctx["npex"] and ctx["npey"]:
            t = _sub_token(t, "<npey.npex>", ctx["npex"] * ctx["npey"])
        if child_ctx["npex"] and child_ctx["npey"]:
            t = _sub_token(t, "<N02_npey.N02.npex>",
                           child_ctx["npex"] * child_ctx["npey"])
        llx = c.origin_x - s.origin_x
        lly = c.origin_y - s.origin_y
        t = _sub_line_positional(t, "'parent',", ["0.0", "0.0"], suffix=".0")
        t = _sub_line_positional(t, "'child',",
                                 [f"{llx:.1f}", f"{lly:.1f}"], suffix=".0")

    # child (N02) header: offset from root + GSR
    if o is not None and s.name == "child":
        llx = s.origin_x - o.origin_x
        lly = s.origin_y - o.origin_y
        t = _sub_line_positional(t, "Offset from root SW corner",
                                 [f"{llx:.1f}", f"{lly:.1f}"])
        t = _sub_line_positional(t, "Grid Spacing Ratio",
                                 [_fmt_g(o.grid_size / s.grid_size)])

    # standalone header: SW corner
    t = _sub_line_positional(t, "SW corner (UTM)",
                             [f"{s.origin_x:.1f}", f"{s.origin_y:.1f}"])

    # model grid
    t = _sub_keyed(t, "nx", s.nx)
    t = _sub_keyed(t, "ny", s.ny)
    if ctx["nz"]:
        t = _sub_keyed(t, "nz", ctx["nz"])
    for k in ("dx", "dy", "dz"):
        t = _sub_keyed(t, k, f"{float(s.grid_size):.1f}", suffix=".0")

    # time origin / run length
    if ctx["origin_time"]:
        t = _sub_keyed(t, "origin_date_time", ctx["origin_time"], sep=r"=\s*'")
    if ctx["hours"]:
        t = _sub_token(t, "<time_in_seconds>", int(float(ctx["hours"]) * 3600))
        t = _sub_token(t, "<time>", _fmt_g(ctx["hours"]))

    # processor topology (namelist lines and the comment line)
    if ctx["npex"] and ctx["npey"]:
        t = _sub_token(t, "<npex.npey>", ctx["npex"] * ctx["npey"])
        t = _sub_keyed(t, "npex", ctx["npex"])
        t = _sub_keyed(t, "npey", ctx["npey"])

    # seasonal soil temperatures (from origin_time month; no-op when the
    # seasonal soil tables are not configured)
    seasonal = ctx.get("seasonal") or {}
    if seasonal.get("soil"):
        vals = ", ".join(f"{v:.2f}" for v in seasonal["soil"])
        t = re.sub(r"(?m)^(\s*soil_temperature\s*=\s*).*$",
                   lambda m: f"{m.group(1)}{vals},", t)
    if seasonal.get("deep_soil") is not None:
        t = re.sub(r"(?m)^(\s*deep_soil_temperature\s*=\s*).*$",
                   lambda m: f"{m.group(1)}{seasonal['deep_soil']:.2f},", t)
    if seasonal.get("soil_moisture"):
        mvals = ", ".join(f"{v:.2f}" for v in seasonal["soil_moisture"])
        t = re.sub(r"(?m)^(\s*soil_moisture\s*=\s*).*$",
                   lambda m: f"{m.group(1)}{mvals},", t)
    for param, k in (seasonal.get("p3d") or {}).items():
        t = re.sub(rf"(?m)^(\s*{re.escape(param)}\s*=\s*)[0-9.]+",
                   lambda m, k=k: f"{m.group(1)}{k:.2f}", t)
    return t


def fill_pgem(text, ctx):
    s = ctx["spec"]
    t = _sub_token(text, "<case>", ctx["case"])
    if ctx.get("hpc_user"):
        t = _sub_token(t, "<hpc_user>", ctx["hpc_user"])
    # The PostgreSQL password is never stored in the repository. It is read
    # from the PALM_PGEM_PASSWORD environment variable; if unset, the
    # <pg_password> placeholder is left in place for the user to fill in.
    pg_password = os.environ.get("PALM_PGEM_PASSWORD")
    if pg_password:
        t = _sub_token(t, "<pg_password>", pg_password)
    t = _sub_keyed(t, "dx", _fmt_g(s.grid_size), sep=":")
    t = _sub_keyed(t, "dy", _fmt_g(s.grid_size), sep=":")
    t = _sub_keyed(t, "dz", _fmt_g(s.grid_size), sep=":")
    t = _sub_keyed(t, "nx", s.nx, sep=":")
    t = _sub_keyed(t, "ny", s.ny, sep=":")
    t = _sub_keyed(t, "cent_x", f"{(s.origin_x + s.maxx) / 2:.3f}", sep=":")
    t = _sub_keyed(t, "cent_y", f"{(s.origin_y + s.maxy) / 2:.3f}", sep=":")
    if ctx["origin_time"]:
        t = _sub_keyed(t, "origin_time", ctx["origin_time"], sep=r":\s*'")
    seasonal = ctx.get("seasonal") or {}
    if seasonal.get("water"):
        t = _apply_water_temps(t, seasonal["water"])
    return t


def fill_pmeteo(text, ctx):
    s = ctx["spec"]
    t = _sub_token(text, "<case>", ctx["case"])
    if ctx.get("wrf_dir"):
        t = _sub_token(t, "<wrf_dir>", ctx["wrf_dir"])
    if ctx["wrf_date"]:
        t = _sub_token(t, "<prague_date>", ctx["wrf_date"])
    t = t.replace("<domain_resolution>.0", f"{float(s.grid_size):.1f}")
    if ctx["nz"]:
        t = _sub_keyed(t, "nz", ctx["nz"], sep=":")
    if ctx["origin_time"]:
        # Unquoted: palm_meteo parses origin_time as a YAML timestamp and
        # errors on the quoted (string) form. The pmeteo templates carry a
        # bare "origin_time: <>", unlike the p3d/pgem templates which supply
        # their own quotes around the placeholder.
        t = _sub_keyed(t, "origin_time", ctx["origin_time"], sep=":")
    if ctx["hours"]:
        t = _sub_keyed(t, "length", _fmt_g(ctx["hours"]), sep=":")
    return t


# ------------------------------
# 4. P3D -> P3DR
# ------------------------------
def make_p3dr(p3d_text):
    """Swap initialization for restart runs: comment the active
    initializing_actions line, activate the read_restart_data one."""
    out = []
    for line in p3d_text.splitlines(keepends=True):
        stripped = line.lstrip()
        if (stripped.startswith("initializing_actions")
                and "read_restart_data" not in stripped):
            out.append("!" + line)
        elif (stripped.startswith("!") and "initializing_actions" in stripped
                and "read_restart_data" in stripped):
            out.append(line.replace("!", " ", 1))
        else:
            out.append(line)
    return "".join(out)


# ------------------------------
# 5. WRITER
# ------------------------------
def _write(out_path, text, overwrite):
    if out_path.exists() and not overwrite:
        log.debug(f"{out_path.name}: exists, kept")
        return
    out_path.write_text(text)
    left = len(_PLACEHOLDER_RE.findall(text)) + text.count("<>")
    note = f" ({left} placeholder(s) left for manual fill)" if left else ""
    log.debug(f"written: {out_path.name}{note}")


def write_templates(child, parent, cfg, state=None):
    """Instantiate all templates for the case into the output directory."""
    tcfg = cfg["templates"]
    tdir = Path(tcfg["dir"]) if tcfg["dir"] else BUILTIN_DIR
    if not tdir.is_dir():
        raise RuntimeError(f"templates.dir not found: {tdir}")
    out_dir = cfg.output_dir
    overwrite = bool(tcfg.get("overwrite", False)
                     or cfg["project"].get("overwrite", False))

    def read(name):
        p = tdir / name
        if not p.exists():
            raise RuntimeError(f"template missing: {p}")
        return p.read_text()

    nested = tcfg.get("nested", True)
    case = tcfg["case"] or cfg["project"]["name"]
    targets = ([f"{case}_p3d", f"{case}_p3d_N02", f"{case}_p3dr",
                f"{case}_p3dr_N02", f"pgem_{case}.yaml",
                f"pgem_{case}_N02.yaml", f"pmeteo_{case}.yaml",
                f"pmeteo_{case}_N02.yaml"] if nested else
               [f"{case}_p3d", f"{case}_p3dr",
                f"pgem_{case}.yaml", f"pmeteo_{case}.yaml"])
    if not overwrite and all((out_dir / t).exists() for t in targets):
        log.debug("all generated files exist, nothing to do "
                  "(templates.overwrite: true to regenerate)")
        return

    _resolve_topology(child, parent, cfg, nested, state)

    progress("Writing run configuration files ({} files)",
             len(targets))
    if nested:
        pctx = build_context(parent, cfg, other=child)
        cctx = build_context(child, cfg, other=parent)
        case = pctx["case"]

        # templates.cyclic switches the p3d skeletons to the cyclic-
        # boundary variants (no offline nesting, flow driven internally)
        var = "cyclic_" if tcfg.get("cyclic") else ""
        if var and not _WARNED.get("cyclic"):
            _WARNED["cyclic"] = True
            log.warning("templates.cyclic: the cyclic p3d templates are a "
                        "STARTING SKELETON (TODO) - review the flow driving "
                        "(ug/vg_surface, dp_external/dpdxy), y_shift and "
                        "disturbance settings before a production run.")
        p3d_parent = fill_p3d(read(f"template_{var}nested_p3d"), pctx,
                              child_ctx=cctx)
        p3d_child = fill_p3d(read(f"template_{var}nested_p3d_N02"), cctx)
        _write(out_dir / f"{case}_p3d", p3d_parent, overwrite)
        _write(out_dir / f"{case}_p3d_N02", p3d_child, overwrite)
        _write(out_dir / f"{case}_p3dr", make_p3dr(p3d_parent), overwrite)
        _write(out_dir / f"{case}_p3dr_N02", make_p3dr(p3d_child), overwrite)

        _write(out_dir / f"pgem_{case}.yaml",
               fill_pgem(read("pgem_template_nested.yaml"), pctx), overwrite)
        _write(out_dir / f"pgem_{case}_N02.yaml",
               fill_pgem(read("pgem_template_nested_N02.yaml"), cctx), overwrite)
        _write(out_dir / f"pmeteo_{case}.yaml",
               fill_pmeteo(read("pmeteo_template_nested.yaml"), pctx), overwrite)
        _write(out_dir / f"pmeteo_{case}_N02.yaml",
               fill_pmeteo(read("pmeteo_template_nested_N02.yaml"), cctx), overwrite)
        nz_resolved = {"parent": pctx["nz"], "child": cctx["nz"]}
    else:
        spec = parent if tcfg.get("single_domain", "parent") == "parent" else child
        ctx = build_context(spec, cfg)
        case = ctx["case"]
        nz_resolved = {spec.name: ctx["nz"]}
        var = "cyclic_" if tcfg.get("cyclic") else ""
        if var and not _WARNED.get("cyclic"):
            _WARNED["cyclic"] = True
            log.warning("templates.cyclic: the cyclic p3d template is a "
                        "STARTING SKELETON (TODO) - review the flow driving "
                        "(ug/vg_surface, dp_external/dpdxy), y_shift and "
                        "disturbance settings before a production run.")
        p3d = fill_p3d(read(f"template_{var}p3d"), ctx)
        _write(out_dir / f"{case}_p3d", p3d, overwrite)
        _write(out_dir / f"{case}_p3dr", make_p3dr(p3d), overwrite)
        _write(out_dir / f"pgem_{case}.yaml",
               fill_pgem(read("pgem_template.yaml"), ctx), overwrite)
        _write(out_dir / f"pmeteo_{case}.yaml",
               fill_pmeteo(read("pmeteo_template.yaml"), ctx), overwrite)

    missing = [k for k in ("origin_time", "length", "wrf_date",
                           "hpc_user", "wrf_dir")
               if not (tcfg["values"].get(k)
                       or (k == "length" and tcfg["values"].get("simulation_hours")))]
    # nz_parent/nz_child are not required: they auto-calculate from the final
    # buildings raster (child) or a fixed target height (parent) unless set
    # explicitly. Only warn about nz when it was neither set nor successfully
    # auto-calculated (nz_resolved holds the value that actually went into
    # each domain's template; None means _auto_nz could not determine it and
    # already warned separately). This is why nz is checked against the built
    # context, not against the raw config keys.
    nz_unresolved = [n for n, v in nz_resolved.items() if not v]
    if nz_unresolved:
        missing.append("nz_" + "/nz_".join(nz_unresolved))
    if missing:
        log.warning(f"values not set in templates.values "
                    f"({', '.join(missing)}); corresponding placeholders "
                    f"remain for manual filling.")
