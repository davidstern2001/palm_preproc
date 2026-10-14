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

import json
import math
import os
import re
import shutil
from datetime import datetime
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
    line, keeping everything around it.

    The key is anchored on its left with a negative lookbehind, so a short
    key can never match the TAIL of a longer one: without it,
    `_sub_keyed(t, "nz", ...)` also fires inside `max_nz = <>` and
    `_sub_keyed(t, "dz", ...)` inside `ref_dz = <>`, silently writing the
    wrong value into a namelist. The current skeletons happen to have no
    such pair, but adding one would corrupt a run with no error anywhere.
    """
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_])({re.escape(key)}\s*{sep}\s*)<>{re.escape(suffix)}")
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


def fmt_real(v):
    """A real for a namelist or YAML: every significant digit, and at least
    one decimal (1 -> '1.0', 0.25 -> '0.25'). Grid spacings and offsets
    used to go through ':.1f', which wrote dx = 0.25 as '0.2'."""
    s = f"{float(v):.10g}"
    return s if any(c in s for c in ".eE") else s + ".0"


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


def _abs_rooftop_heights(cfg, domain_name):
    """(mean, max) absolute rooftop height over the domain's building
    cells, in metres, referenced to the domain's lowest terrain point (the
    datum PALM's static driver uses: min(zt) -> z = 0). Or None if it
    can't be determined.

    For each building cell: abs_roof = terrain + building_height - min(terrain
    over the whole domain). Both statistics are taken over building cells
    only, so a hill with no buildings on it raises min(terrain) into the
    datum but does NOT inflate them; buildings sitting on higher ground
    correctly count at their true absolute rooftop height. The mean drives
    the roughness-sublayer target (dz*nz ~ 3-5 x mean building height) and
    the max keeps the tallest building below the model top.

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
        log.warning(f"[templates] {domain_name}: DEM {dem.shape} and buildings "
                    f"{bld.shape} are not grid-aligned - no absolute rooftop "
                    f"height (check clip.snap_rasters_to_grid).")
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
    return float(abs_roof.mean()), float(abs_roof.max())


def child_nz(h_mean, h_max, dz, factor, tallest_factor, stride):
    """Child nz from rooftop heights: the larger of the roughness-sublayer
    target (factor x mean rooftop height) and the clearance over the
    tallest building (tallest_factor x max rooftop height), snapped up to
    the multigrid stride. Returns (nz, raw points, which rule governed).

    The mean alone let one tower among low blocks reach the model top: a
    100 m tower among 15 m buildings gives a mean near 16 m and, at
    factor 4, a 64 m domain."""
    by_mean = factor * h_mean / dz
    by_max = tallest_factor * h_max / dz
    nz_raw = max(by_mean, by_max)
    return snap_up(nz_raw, stride), nz_raw, (
        "mean" if by_mean >= by_max else "tallest")


def _auto_nz(cfg, spec):
    """Default vertical extent (grid points), used when nz_<domain> is not
    set explicitly in templates.values:
      * child:  dz*nz = child_nz_height_factor x mean absolute rooftop height
                (terrain + building, referenced to the domain's lowest terrain
                point) over the FINAL buildings raster -- the roughness-
                sublayer target in the static driver's absolute-height frame
                -- but at least child_nz_tallest_factor x the tallest
                absolute rooftop, so no building reaches the model top.
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
        tallest = float(tv.get("child_nz_tallest_factor", 1.5))
        heights = _abs_rooftop_heights(cfg, spec.name)
        basis = "absolute rooftop height"
        if heights is None:
            # Grids not aligned, or no DEM: fall back to the relative
            # (above-terrain) building height so a driver is still produced.
            # Note this can undersize nz where buildings sit well above the
            # domain's lowest terrain.
            h = _max_building_height(cfg, spec.name)
            heights = None if h is None else (h, h)
            basis = "max relative building height (DEM unavailable/unaligned)"
        if heights is None:
            log.warning(f"[templates] {spec.name}: no building height for "
                        f"nz (buildings raster missing or empty) - nz left "
                        f"as a placeholder.")
            return None
        h_mean, h_max = heights
        # Snapped up to a multiple of `stride` for the multigrid solver.
        nz, nz_raw, rule = child_nz(h_mean, h_max, dz, factor, tallest,
                                    stride)
        if rule == "mean":
            why = f"{factor:g} x mean {basis} {h_mean:.1f} m"
        else:
            why = (f"{tallest:g} x tallest {basis} {h_max:.1f} m, above "
                   f"{factor:g} x mean {h_mean:.1f} m")
        log.info(f"[templates] {spec.name}: nz = {nz} ({why}, "
                 f"{nz_raw:.1f} pts snapped to x{stride})")
        return nz
    else:
        target = float(tv.get("parent_target_height_m", 2000.0))
        nz_raw = target / dz
        nz = snap_up(nz_raw, stride)
        log.info(f"[templates] {spec.name}: nz = {nz} ({target:g} m, "
                 f"{nz_raw:.1f} pts snapped to x{stride})")
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
            log.warning(f"[templates] seasonal: cannot parse {path.name}: "
                        f"{e} - not applied.")
            file_cfg = {}
    return _merge_seasonal(file_cfg, _cfg_get(cfg, "seasonal") or {})


def _seasonal_key(scfg, canonical, deprecated):
    """Read a seasonal table under its canonical name, accepting the old
    name with a deprecation warning."""
    if scfg.get(canonical) is not None:
        return scfg.get(canonical)
    if scfg.get(deprecated) is not None:
        log.warning(f"[templates] seasonal: '{deprecated}' is now "
                    f"'{canonical}' - the old name still works for now.")
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
        log.warning(f"[templates] seasonal: no month in origin_time "
                    f"'{origin_time}' - not applied.")
        return empty
    month_idx = int(m.group(1)) - 1          # 0-based Jan..Dec
    stat = str(scfg.get("statistic", "ave")).lower()
    if stat not in _SEASONAL_STATS:
        log.warning(f"[templates] seasonal.statistic '{stat}' not one of "
                    f"{_SEASONAL_STATS} - using 'ave'.")
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
            got = len(table) if isinstance(table, (list, tuple)) else repr(table)
            log.warning(f"[templates] seasonal: {what} needs 12 monthly "
                        f"values, got {got} - not applied.")
            return None
        return float(table[month_idx])

    out = {"water": {}, "soil": None, "deep_soil": None,
           "soil_moisture": None, "p3d": {}}
    old_keys = [k for k in ("water_temperature_c", "soil_temperature_c",
                            "deep_soil_temperature_c", "p3d_temperature_c")
                if k in scfg]
    if old_keys:
        log.warning(f"[templates] seasonal: old Celsius key(s) {old_keys} "
                    f"ignored - tables are now in KELVIN (*_k).")
    for type_id, table in (_seasonal_key(scfg, "water_pars_temp",
                                     "water_temperature_k") or {}).items():
        k = month_value(table, f"water_pars_temp[{type_id}]")
        if k is not None:
            out["water"][int(type_id)] = k
    ph = set(int(i) for i in (scfg.get("placeholder_water_types") or []))
    used_ph = sorted(ph & set(out["water"]))
    if used_ph and not _WARNED.get("placeholder_water"):
        _WARNED["placeholder_water"] = True   # once per run, not per domain
        # Replace them with measured monthly data in seasonal.yaml and drop
        # the ids from placeholder_water_types.
        log.warning(f"[templates] seasonal: water type(s) {used_ph} still use "
                    f"placeholder tables (TODO).")

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
                log.warning(f"[templates] seasonal: soil_temperature rows "
                            f"need 8 layers, got {len(layer_row)}.")
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
                log.warning(f"[templates] seasonal: soil_moisture rows "
                            f"need 8 layers, got {len(layer_row)}.")
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
    (palm_meteo's duration style). Returns float hours, or None.

    A bare number written as a STRING ('29', or `length: "29"` in YAML,
    which is also what a value like `29.0` becomes once quoted) counts as
    hours too. It used to fall through the unit regex and return None, so
    the length was silently dropped and `<time>` / `<time_in_seconds>` were
    left as placeholders in the p3d.
    """
    if value is None:
        return None
    if isinstance(value, bool):        # YAML `length: yes` -> True
        return None
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    try:
        return float(s)                # bare numeric string = hours
    except ValueError:
        pass
    m = re.match(r"\s*([0-9.]+)\s*(h|hr|hours?|d|days?|m|min|minutes?|s|secs?|seconds?)\s*$",
                 s, re.I)
    if not m:
        log.warning(f"[templates] templates.values.length {value!r} is not "
                    f"a duration (e.g. 29, '29 h', '1 d') - left unset.")
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
        log.warning(f"[templates] {name}: npex/npey not resolved - left as "
                    f"placeholders.")
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


_TOPO_KEYS = ("npex_parent", "npey_parent", "npex_child", "npey_child")


def _check_pins(tv, specs, opts):
    """Validate the npex/npey pinned in templates.values for the domains in
    `specs` ({name: DomainSpec}). A pin that does not divide the domain's
    grid points is an error (PALM aborts on it, after queueing); a pin
    PALM accepts but Hynek Reznicek's rules do not recommend is a warning.
    Pins are normalised to int in place."""
    for name, spec in specs.items():
        for axis, pts in (("npex", spec.width_pts), ("npey", spec.height_pts)):
            key = f"{axis}_{name}"
            v = tv.get(key)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                f = math.nan
            if isinstance(v, bool) or not f.is_integer() or f < 1:
                raise RuntimeError(f"templates.values.{key} = {v!r} is not a "
                                   f"positive whole number.")
            v = tv[key] = int(f)
            if pts % v:
                d = axis[-1]
                raise RuntimeError(
                    f"templates.values.{key} = {v} does not divide the "
                    f"{name} domain's {pts} grid points along {d} (PALM "
                    f"needs n{d}+1 divisible by {axis}); remove the pin or "
                    f"pick a divisor.")
        npex, npey = tv.get(f"npex_{name}"), tv.get(f"npey_{name}")
        if npex and npey and (npex, npey) not in {
                p[:2] for p in valid_pairs(spec.width_pts, spec.height_pts,
                                           opts[name])}:
            log.warning(f"[topology] {name}: pinned npex={npex} npey={npey} "
                        f"is outside the recommended topology rules "
                        f"(PALM accepts it; check the cells per core).")


def _pinned_matches(configs, tv, names, nested):
    """Keep the configurations agreeing with every pinned npex/npey."""
    pins = {(n, a): tv[f"{a}_{n}"] for n in names for a in ("npex", "npey")
            if tv.get(f"{a}_{n}")}
    if not pins:
        return configs, pins
    kept = [c for c in configs
            if all((c[n] if nested else c)[a] == v
                   for (n, a), v in pins.items())]
    log.debug(f"[topology] pinned {pins}: {len(kept)}/{len(configs)} kept")
    return kept, pins


def _resolve_topology(child, parent, cfg, nested, state=None):
    """Fill templates.values npex/npey via Hynek Reznicek's topology rules,
    interactively when configured and possible. Pinned values win: all of
    them pinned bypasses the chooser, some of them pinned restricts it to
    the configurations that agree with them (they used to be overwritten
    silently). The resolved topology is persisted to the pipeline state so
    the report can document the actual choice."""
    tv = cfg["templates"]["values"]
    raw_mode = cfg["templates"].get("topology_select", True)
    mode = "ask" if raw_mode in (True, "ask", "true") else "auto"
    node_cpus = tv.get("node_cpus")
    topo_opt = cfg["domains"]["parent"].get("topology_opt", True)
    topo_opt_child = cfg["domains"]["child"].get("topology_opt", topo_opt)
    opts = {"parent": topo_opt, "child": topo_opt_child}

    progress("Selecting processor topology")
    if nested:
        _check_pins(tv, {"parent": parent, "child": child}, opts)
        if all(tv.get(k) for k in _TOPO_KEYS):
            ratio = (tv["npex_parent"] * tv["npey_parent"]
                     / (tv["npex_child"] * tv["npey_child"]))
            if not 0.300 < ratio < 1.001:
                log.warning(f"[topology] pinned parent:child cores "
                            f"{ratio:.2f} is outside the recommended "
                            f"1:3 .. 1:1.")
            _persist_topology(tv, state)
            return
        configs = nested_configurations(parent, child, topo_opt, topo_opt_child)
        configs, pins = _pinned_matches(configs, tv, ("parent", "child"),
                                        nested=True)
    else:
        name = cfg["templates"].get("single_domain", "parent")
        spec = parent if name == "parent" else child
        _check_pins(tv, {name: spec}, opts)
        if tv.get(f"npex_{name}") and tv.get(f"npey_{name}"):
            _persist_topology(tv, state)
            return
        configs = single_configurations(spec, opts[name])
        configs, pins = _pinned_matches(configs, tv, (name,), nested=False)
    if pins and not configs:
        shown = ", ".join(f"{a}_{n}={v}" for (n, a), v in pins.items())
        raise RuntimeError(
            f"No valid topology agrees with the pinned {shown}. Pin all of "
            f"npex/npey to use exactly those values, or remove the pin(s).")

    configs = filter_by_nodes(configs, node_cpus, tv.get("min_nodes"),
                              tv.get("max_nodes"), nested=nested)
    configs = dedupe_by_cores(configs, nested=nested)
    if not cfg["templates"].get("list_all_topologies", False):
        configs = best_per_node_group(configs, node_cpus, nested=nested)
    if nested:
        rec = recommend_nested(configs, tv.get("npes_parent"),
                               tv.get("npes_child"), node_cpus)
    else:
        rec = recommend_single(configs, tv.get(f"npes_{name}"), node_cpus)
    idx = choose_topology(configs, rec, mode, node_cpus, nested=nested)
    if idx is None:
        return
    chosen = configs[idx]
    if nested:
        tv["npex_parent"], tv["npey_parent"] = (chosen["parent"]["npex"],
                                                chosen["parent"]["npey"])
        tv["npex_child"], tv["npey_child"] = (chosen["child"]["npex"],
                                              chosen["child"]["npey"])
        log.info(f"[topology] parent npex={tv['npex_parent']} "
                 f"npey={tv['npey_parent']} ({chosen['parent']['cores']}), "
                 f"N02 npex={tv['npex_child']} npey={tv['npey_child']} "
                 f"({chosen['child']['cores']}), sum_cpu={chosen['sum']}")
    else:
        tv[f"npex_{name}"], tv[f"npey_{name}"] = chosen["npex"], chosen["npey"]
        log.info(f"[topology] {name} npex={chosen['npex']} "
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
                                 [fmt_real(llx), fmt_real(lly)], suffix=".0")

    # child (N02) header: offset from root + GSR
    if o is not None and s.name == "child":
        llx = s.origin_x - o.origin_x
        lly = s.origin_y - o.origin_y
        t = _sub_line_positional(t, "Offset from root SW corner",
                                 [fmt_real(llx), fmt_real(lly)])
        t = _sub_line_positional(t, "Grid Spacing Ratio",
                                 [_fmt_g(o.grid_size / s.grid_size)])

    # standalone header: SW corner
    t = _sub_line_positional(t, "SW corner (UTM)",
                             [fmt_real(s.origin_x), fmt_real(s.origin_y)])

    # model grid
    t = _sub_keyed(t, "nx", s.nx)
    t = _sub_keyed(t, "ny", s.ny)
    if ctx["nz"]:
        t = _sub_keyed(t, "nz", ctx["nz"])
    for k in ("dx", "dy", "dz"):
        t = _sub_keyed(t, k, fmt_real(s.grid_size), suffix=".0")

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
    # PALM-GeM's nx/ny are the NUMBER of grid points, unlike the _p3d,
    # where nx/ny are the last index (points - 1). Writing s.nx here made
    # the static driver one column and one row short, and PALM aborted in
    # topography setup (PAC0333: static driver x = nx, model expects nx+1).
    t = _sub_keyed(t, "nx", s.width_pts, sep=":")
    t = _sub_keyed(t, "ny", s.height_pts, sep=":")
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
    t = t.replace("<domain_resolution>.0", fmt_real(s.grid_size))
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
def _write(out_path, text):
    out_path.write_text(text)
    # The pattern matches the bare `<>` too, so this counts each once (it
    # used to add text.count("<>") on top and report them twice).
    left = len(_PLACEHOLDER_RE.findall(text))
    note = f" ({left} placeholder(s) to fill)" if left else ""
    log.info(f"[templates] wrote {out_path.name}{note}")


def fill_submit(text, cfg, case, total_cores):
    """Fill the palmrun submit script. -X (total MPI processes) comes from
    the topology actually chosen for this case, -T from node_cpus; the rest
    are cluster/run choices taken from templates.values. Unset values leave
    their <> placeholders, as everywhere else."""
    tv = cfg["templates"]["values"]
    t = _sub_token(text, "<case>", case)
    t = _sub_token(t, "<palmrun_id>", tv.get("palmrun_id") or case)
    if total_cores:
        t = _sub_token(t, "<total_cores>", total_cores)
    if tv.get("node_cpus"):
        t = _sub_token(t, "<node_cpus>", tv["node_cpus"])
    # Wall-clock limit: same duration syntax as templates.values.length
    # ("48 h", "2 d", or a bare number of hours), written out in seconds
    # because that is what palmrun's -t expects.
    hours = _parse_hours(tv.get("walltime"))
    if hours:
        t = _sub_token(t, "<walltime_s>", int(round(hours * 3600)))
    for key, token in (("palmrun_config", "<palmrun_config>"),
                       ("palmrun_activation", "<palmrun_activation>"),
                       ("queue", "<queue>")):
        if tv.get(key):
            t = _sub_token(t, token, tv[key])
    return t


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
    single_name = tcfg.get("single_domain", "parent")
    specs = ({"parent": parent, "child": child} if nested else
             {single_name: parent if single_name == "parent" else child})
    targets = ([f"{case}_p3d", f"{case}_p3d_N02", f"{case}_p3dr",
                f"{case}_p3dr_N02", f"pgem_{case}.yaml",
                f"pgem_{case}_N02.yaml", f"pmeteo_{case}.yaml",
                f"pmeteo_{case}_N02.yaml"] if nested else
               [f"{case}_p3d", f"{case}_p3dr",
                f"pgem_{case}.yaml", f"pmeteo_{case}.yaml"])
    write_submit = bool(tcfg.get("submit", True))
    if write_submit:
        targets.append(f"submit_{case}.sh")
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = _plan_set(out_dir, case, targets, specs, nested, single_name,
                     tcfg["values"], overwrite, state)
    if not todo:
        return

    _resolve_topology(child, parent, cfg, nested, state)

    def put(name, text):
        if name in todo:
            _write(out_dir / name, text)

    progress("Writing run configuration files ({} files)", len(todo))
    # templates.cyclic switches the p3d skeletons to the cyclic-boundary
    # variants (no offline nesting, flow driven internally)
    var = "cyclic_" if tcfg.get("cyclic") else ""
    if var and not _WARNED.get("cyclic"):
        _WARNED["cyclic"] = True
        # Review the flow driving (ug/vg_surface, dp_external/dpdxy),
        # y_shift and disturbances before a production run.
        log.warning("[templates] cyclic p3d is a STARTING SKELETON "
                    "(TODO) - review flow driving and y_shift.")
    if nested:
        pctx = build_context(parent, cfg, other=child)
        cctx = build_context(child, cfg, other=parent)
        case = pctx["case"]
        p3d_parent = fill_p3d(read(f"template_{var}nested_p3d"), pctx,
                              child_ctx=cctx)
        p3d_child = fill_p3d(read(f"template_{var}nested_p3d_N02"), cctx)
        put(f"{case}_p3d", p3d_parent)
        put(f"{case}_p3d_N02", p3d_child)
        put(f"{case}_p3dr", make_p3dr(p3d_parent))
        put(f"{case}_p3dr_N02", make_p3dr(p3d_child))
        put(f"pgem_{case}.yaml",
            fill_pgem(read("pgem_template_nested.yaml"), pctx))
        put(f"pgem_{case}_N02.yaml",
            fill_pgem(read("pgem_template_nested_N02.yaml"), cctx))
        put(f"pmeteo_{case}.yaml",
            fill_pmeteo(read("pmeteo_template_nested.yaml"), pctx))
        put(f"pmeteo_{case}_N02.yaml",
            fill_pmeteo(read("pmeteo_template_nested_N02.yaml"), cctx))
        nz_resolved = {"parent": pctx["nz"], "child": cctx["nz"]}
    else:
        spec = specs[single_name]
        ctx = build_context(spec, cfg)
        case = ctx["case"]
        nz_resolved = {spec.name: ctx["nz"]}
        p3d = fill_p3d(read(f"template_{var}p3d"), ctx)
        put(f"{case}_p3d", p3d)
        put(f"{case}_p3dr", make_p3dr(p3d))
        put(f"pgem_{case}.yaml", fill_pgem(read("pgem_template.yaml"), ctx))
        put(f"pmeteo_{case}.yaml",
            fill_pmeteo(read("pmeteo_template.yaml"), ctx))

    # -- palmrun submit script ----------------------------------------
    # -X must equal the TOTAL number of MPI processes: for a nested run that
    # is parent + child, not either one alone.
    tv = tcfg["values"]
    if write_submit:
        total_cores = sum((tv.get(f"npex_{n}") or 0) * (tv.get(f"npey_{n}") or 0)
                          for n in specs)
        sub_path = out_dir / f"submit_{case}.sh"
        put(sub_path.name, fill_submit(read("submit_template.sh"), cfg, case,
                                       total_cores or None))
        if sub_path.exists():
            sub_path.chmod(sub_path.stat().st_mode | 0o111)   # make executable
    _write_record(out_dir, case, nested, single_name, specs, tv, nz_resolved)

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
        log.warning(f"[templates] not set in templates.values: "
                    f"{', '.join(missing)} - placeholders left.")


# ------------------------------
# 6. THE GENERATED SET: WHAT IS KEPT, WHAT IS (RE)WRITTEN
# ------------------------------
# Generated files are never overwritten (they may carry hand edits), but a
# kept file must still describe THIS case. Each set records the geometry,
# topology and nz it was written for in palm_preproc_templates_<case>.json;
# sets written before the record existed are read back from their own
# _p3d and pgem files instead.

def _record_path(out_dir, case):
    return out_dir / f"palm_preproc_templates_{case}.json"


def _geometry(spec):
    return {"grid_size": float(spec.grid_size),
            "origin_x": round(float(spec.origin_x), 6),
            "origin_y": round(float(spec.origin_y), 6),
            "width_pts": int(spec.width_pts),
            "height_pts": int(spec.height_pts)}


def _uncommented(path, mark):
    return "\n".join(line.split(mark, 1)[0]
                     for line in path.read_text().splitlines())


def _p3d_values(path):
    """nx, ny, nz, npex, npey and - in a nested parent - the child's
    llx/lly, read back from a written _p3d (None where absent or still a
    placeholder)."""
    text = _uncommented(path, "!")
    out = {}
    for key in ("nx", "ny", "nz", "npex", "npey"):
        m = re.search(rf"(?<![A-Za-z0-9_]){key}\s*=\s*(\d+)\b", text)
        out[key] = int(m.group(1)) if m else None
    m = re.search(r"domain_layouts\s*=[^/]*?,\s*2\s*,\s*1\s*,\s*[^,]+,"
                  r"\s*(-?[\d.]+)\s*,\s*(-?[\d.]+)", text)
    out["llx_lly"] = (float(m.group(1)), float(m.group(2))) if m else None
    return out


def _pgem_values(path):
    """nx, ny (int) and cent_x, cent_y (float) read back from a pgem yaml."""
    text = _uncommented(path, "#")
    out = {}
    for key, kind in (("nx", int), ("ny", int),
                      ("cent_x", float), ("cent_y", float)):
        m = re.search(rf"(?m)^\s*{key}\s*:\s*(-?[\d.]+)\s*$", text)
        out[key] = kind(m.group(1)) if m else None
    return out


def _set_files(case, nested, single_name):
    """{domain: (p3d name, pgem name)} of a set."""
    if nested:
        return {"parent": (f"{case}_p3d", f"pgem_{case}.yaml"),
                "child": (f"{case}_p3d_N02", f"pgem_{case}_N02.yaml")}
    return {single_name: (f"{case}_p3d", f"pgem_{case}.yaml")}


def _read_record(out_dir, case, nested, single_name):
    """The set's record, or what the existing files tell (legacy sets);
    None when there is nothing to go on."""
    path = _record_path(out_dir, case)
    if path.exists():
        try:
            return json.loads(path.read_text())
        except ValueError as exc:
            log.warning(f"[templates] {path.name} unreadable ({exc}) - "
                        f"checking the files themselves.")
    rec = {"legacy": True, "domains": {}, "topology": {}, "nz": {}}
    for name, (p3d, pgem) in _set_files(case, nested, single_name).items():
        dom = {}
        if (out_dir / p3d).exists():
            v = _p3d_values(out_dir / p3d)
            if v["nx"] is not None and v["ny"] is not None:
                dom["width_pts"], dom["height_pts"] = v["nx"] + 1, v["ny"] + 1
            if v["nz"]:
                rec["nz"][name] = v["nz"]
            for axis in ("npex", "npey"):
                if v[axis]:
                    rec["topology"][f"{axis}_{name}"] = v[axis]
            if nested and name == "parent" and v["llx_lly"]:
                rec["llx_lly"] = v["llx_lly"]
        if (out_dir / pgem).exists():
            v = _pgem_values(out_dir / pgem)
            if v["cent_x"] is not None and v["cent_y"] is not None:
                dom["centre"] = (v["cent_x"], v["cent_y"])
        if dom:
            rec["domains"][name] = dom
    return rec if (rec["domains"] or rec["topology"] or rec["nz"]) else None


def _stale_reason(rec, specs, nested, single_name):
    """Why the existing set no longer describes this case, or None."""
    if not rec.get("legacy"):
        if rec.get("nested") != nested or (
                not nested and rec.get("single_domain") != single_name):
            return "nested / single_domain changed"
        for name, spec in specs.items():
            old, new = rec.get("domains", {}).get(name) or {}, _geometry(spec)
            if any(k not in old or abs(float(old[k]) - v) > 1e-6
                   for k, v in new.items()):
                return f"{name} domain geometry changed"
        return None
    # Files written before the record existed: compare what they say.
    for name, spec in specs.items():
        d = rec["domains"].get(name, {})
        if "width_pts" in d and (d["width_pts"], d["height_pts"]) != (
                spec.width_pts, spec.height_pts):
            return (f"{name} grid {d['width_pts']}x{d['height_pts']} -> "
                    f"{spec.width_pts}x{spec.height_pts}")
        if "centre" in d:
            cx, cy = (spec.origin_x + spec.maxx) / 2, (spec.origin_y + spec.maxy) / 2
            if max(abs(d["centre"][0] - cx), abs(d["centre"][1] - cy)) > 0.0011:
                return f"{name} domain moved"
    if nested and rec.get("llx_lly"):
        llx = specs["child"].origin_x - specs["parent"].origin_x
        lly = specs["child"].origin_y - specs["parent"].origin_y
        # written with one decimal before 1.4
        if max(abs(rec["llx_lly"][0] - llx), abs(rec["llx_lly"][1] - lly)) > 0.051:
            return "child offset in the parent changed"
    return None


def _pgem_points_stale(path, spec):
    """Why a pgem yaml's nx/ny don't match the domain, or None. Before
    1.3.1 pgem got the _p3d's nx (points - 1), which PALM-GeM reads as a
    point count, so those static drivers came out one column short."""
    v = _pgem_values(path)
    if v["nx"] is None or v["ny"] is None:
        return None
    if (v["nx"], v["ny"]) == (spec.width_pts, spec.height_pts):
        return None
    if (v["nx"], v["ny"]) == (spec.width_pts - 1, spec.height_pts - 1):
        return "pgem nx/ny written as points - 1 (palm_preproc < 1.3.1)"
    return f"pgem nx/ny {v['nx']}x{v['ny']} != {spec.width_pts}x{spec.height_pts}"


def _adopt_recorded(tv, rec, specs):
    """Files written now must match the kept ones: take their topology and
    nz, warning where the config asks for something else."""
    wanted = {k: v for k, v in rec.get("topology", {}).items()
              if k.rsplit("_", 1)[1] in specs}
    wanted.update({f"nz_{n}": v for n, v in rec.get("nz", {}).items()
                   if n in specs})
    for key, val in wanted.items():
        cur = tv.get(key)
        if cur is not None and cur != val:
            log.warning(f"[templates] templates.values.{key} = {cur} but the "
                        f"existing files use {val}; they are kept, so the "
                        f"files written now use {val} too. Delete the set "
                        f"to regenerate it with {cur}.")
        tv[key] = val


def _write_record(out_dir, case, nested, single_name, specs, tv, nz):
    from .. import __version__
    rec = {"palm_preproc": __version__, "case": case, "nested": nested,
           "single_domain": None if nested else single_name,
           "domains": {n: _geometry(s) for n, s in specs.items()},
           "topology": {k: tv[k] for k in _TOPO_KEYS
                        if tv.get(k) and k.rsplit("_", 1)[1] in specs},
           "nz": {n: v for n, v in nz.items() if v}}
    _record_path(out_dir, case).write_text(json.dumps(rec, indent=2) + "\n")


def _plan_set(out_dir, case, targets, specs, nested, single_name, tv,
              overwrite, state):
    """Which files of the set to (re)write. Existing files are kept unless
    they no longer describe this case: then they are moved to
    _superseded/<time>/ and rewritten, with a warning (they used to be kept
    with an info line, so a changed buffer left _p3d files with the old
    nx/ny and child offset next to data for the new domains). Files that
    are written next to kept ones reuse the kept ones' topology and nz."""
    existing = [t for t in targets if (out_dir / t).exists()]
    if overwrite or not existing:
        return list(targets)
    rec = _read_record(out_dir, case, nested, single_name)
    reason = _stale_reason(rec, specs, nested, single_name) if rec else None
    stale = dict.fromkeys(existing, reason) if reason else {}
    if not reason:
        for name, (_, pgem) in _set_files(case, nested, single_name).items():
            if pgem in existing:
                why = _pgem_points_stale(out_dir / pgem, specs[name])
                if why:
                    stale[pgem] = why
    keep = [t for t in existing if t not in stale]
    for t in keep:
        log.info(f"[templates] kept (exists): {t}")
    if rec and not reason and keep:
        _adopt_recorded(tv, rec, specs)
    if stale:
        backup = (out_dir / "_superseded"
                  / datetime.now().strftime("%Y%m%d-%H%M%S"))
        backup.mkdir(parents=True, exist_ok=True)
        for t in stale:
            shutil.move(str(out_dir / t), str(backup / t))
        log.warning(f"[templates] {len(stale)} existing file(s) no longer "
                    f"match this case ({next(iter(stale.values()))}) - "
                    f"rewritten; the old ones are in "
                    f"{backup.relative_to(out_dir)}/")
    write = [t for t in targets if t not in keep]
    if not write:
        log.info("[templates] kept (all files exist)")
        _persist_topology(tv, state)
        if rec and rec.get("legacy"):
            _write_record(out_dir, case, nested, single_name, specs, tv,
                          {n: tv.get(f"nz_{n}") for n in specs})
    return write
