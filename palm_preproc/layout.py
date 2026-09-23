"""The newcomer config layout, translated onto the internal settings.

A config in this layout says where the data is (`input`), what the two
domains are (`domains`), what run the templates describe (`run`,
`cluster`) and what is written (`templates`, `report`). Rarely needed
settings live in `advanced`. See template.yaml.

Everything downstream still works on the internal structure (`crs`,
`user_data`, `raw_data`, `templates.values`, ...), which is also what the
pre-1.3 layout writes directly. This module only maps one onto the other,
and checks the new layout strictly: an unknown key is an error with a
suggestion, because a silently ignored typo is the commonest way a config
does not do what it says.

The same section names and the same vocabulary as palm2gis and
palm_postproc: `project`, `input`, `run`, `advanced`.

Public API
----------
  is_new_layout(raw)  ->  bool
  translate(raw)      ->  (legacy-style dict, notes)
"""

import copy
import difflib
import re

# ------------------------------
# 1. SCHEMA
# ------------------------------
_LAYERS = {"dem": None, "buildings": None, "landcover": None,
           "roofs": None, "walls": None, "trees": None}

SCHEMA = {
    "project": {"name": None, "root": None, "output_dir": None,
                "overwrite": None, "defaults": None, "state_file": None},
    "input": {"crs": None, "domain_crs": None, "aux_files": None,
              "raw_data": {"dir": None, "layers": dict(_LAYERS)},
              "user_data": {"dir": None, "domain": None, "domain_from": None,
                            "layers": dict(_LAYERS)}},
    "domains": {"child": {"grid_size": None, "buffer": None, "nz": None},
                "parent": {"grid_size": None, "buffer": None, "nz": None}},
    "run": {"origin_time": None, "length": None, "wrf_date": None},
    "cluster": {"user": None, "wrf_dir": None, "queue": None,
                "walltime": None, "node_cpus": None, "nodes": None,
                "cores": None, "palmrun_id": None, "palmrun_config": None,
                "palmrun_activation": None},
    "templates": {"enabled": None, "case": None, "nested": None,
                  "single_domain": None, "submit": None, "cyclic": None},
    "report": {"enabled": None, "file": None, "parent_name": None,
               "child_name": None},
    "seasonal": None,            # deep-merged over defaults/templates/seasonal.yaml
    "advanced": {
        "stages": None, "templates_dir": None,
        "grid": {"nz_factor": None, "parent_target_height": None,
                 "min_power": None, "align_child_to_parent": None,
                 "strict_nesting": None},
        "cpu_topology": {"select": None, "list_all": None,
                         "optimize_child": None, "optimize_parent": None,
                         "confirm_optimized": None, "max_overhead": None,
                         "cells_per_core_div4": None,
                         "npex_parent": None, "npey_parent": None,
                         "npex_child": None, "npey_child": None},
        "clip": None,            # passed through to the internal clip section
        "merge": None,
        "boundary_cleanup": None,
    },
}

# Top-level keys of the pre-1.3 layout, and where each went.
_LEGACY_HINT = {
    "crs": "input.crs and input.domain_crs",
    "user_data": "input.user_data",
    "raw_data": "input.raw_data (and input.aux_files)",
    "clip": "advanced.clip",
    "merge": "advanced.merge",
    "boundary_cleanup": "advanced.boundary_cleanup",
    "stages": "advanced.stages",
    "inputs": "input.user_data",
}

# new path -> internal path, for the settings that map one to one.
_DIRECT = [
    ("project.name", "project.name"),
    ("project.root", "project.root"),
    ("project.output_dir", "project.output_dir"),
    ("project.overwrite", "project.overwrite"),
    ("project.state_file", "project.state_file"),
    ("input.crs", "crs.aligned"),
    ("input.domain_crs", "crs.domain_output"),
    ("input.raw_data.dir", "raw_data.dir"),
    ("input.raw_data.layers", "raw_data.layers"),
    ("input.aux_files", "raw_data.aux_files"),
    ("input.user_data.dir", "user_data.dir"),
    ("input.user_data.domain", "user_data.domain"),
    ("input.user_data.domain_from", "user_data.domain_from"),
    ("input.user_data.layers", "user_data.layers"),
    ("domains.child.grid_size", "domains.child.grid_size"),
    ("domains.child.buffer", "domains.child.buffer"),
    ("domains.parent.grid_size", "domains.parent.grid_size"),
    ("domains.parent.buffer", "domains.parent.buffer"),
    ("domains.child.nz", "templates.values.nz_child"),
    ("domains.parent.nz", "templates.values.nz_parent"),
    ("run.origin_time", "templates.values.origin_time"),
    ("run.length", "templates.values.length"),
    ("run.wrf_date", "templates.values.wrf_date"),
    ("cluster.user", "templates.values.hpc_user"),
    ("cluster.wrf_dir", "templates.values.wrf_dir"),
    ("cluster.queue", "templates.values.queue"),
    ("cluster.walltime", "templates.values.walltime"),
    ("cluster.node_cpus", "templates.values.node_cpus"),
    ("cluster.palmrun_id", "templates.values.palmrun_id"),
    ("cluster.palmrun_config", "templates.values.palmrun_config"),
    ("cluster.palmrun_activation", "templates.values.palmrun_activation"),
    ("templates.case", "templates.case"),
    ("templates.nested", "templates.nested"),
    ("templates.single_domain", "templates.single_domain"),
    ("templates.submit", "templates.submit"),
    ("templates.cyclic", "templates.cyclic"),
    ("report.file", "report.file"),
    ("report.parent_name", "report.parent_name"),
    ("report.child_name", "report.child_name"),
    ("seasonal", "seasonal"),
    ("advanced.stages", "stages"),
    ("advanced.templates_dir", "templates.dir"),
    ("advanced.clip", "clip"),
    ("advanced.merge", "merge"),
    ("advanced.boundary_cleanup", "boundary_cleanup"),
    ("advanced.grid.nz_factor", "templates.values.child_nz_height_factor"),
    ("advanced.grid.parent_target_height",
     "templates.values.parent_target_height_m"),
    ("advanced.grid.align_child_to_parent", "domains.align_child_to_parent"),
    ("advanced.grid.strict_nesting", "domains.strict_nesting"),
    ("advanced.cpu_topology.select", "templates.topology_select"),
    ("advanced.cpu_topology.list_all", "templates.list_all_topologies"),
    ("advanced.cpu_topology.optimize_child",
     "domains.child.optimize_topology"),
    ("advanced.cpu_topology.optimize_parent",
     "domains.parent.optimize_topology"),
    ("advanced.cpu_topology.confirm_optimized",
     "domains.child.confirm_optimized"),
    ("advanced.cpu_topology.cells_per_core_div4", "domains.child.topology_opt"),
    ("advanced.cpu_topology.npex_parent", "templates.values.npex_parent"),
    ("advanced.cpu_topology.npey_parent", "templates.values.npey_parent"),
    ("advanced.cpu_topology.npex_child", "templates.values.npex_child"),
    ("advanced.cpu_topology.npey_child", "templates.values.npey_child"),
]

# Settings that reach BOTH domains from one key.
_BOTH_DOMAINS = {
    "advanced.grid.min_power": "min_power",
    "advanced.cpu_topology.max_overhead": "topology_max_overhead",
    "advanced.cpu_topology.cells_per_core_div4": "topology_opt",
}


class LayoutError(Exception):
    pass


# ------------------------------
# 2. DETECTION AND KEY CHECKING
# ------------------------------
def is_new_layout(raw):
    """The new layout is recognised by its own top-level blocks."""
    if not isinstance(raw, dict):
        return False
    if any(k in raw for k in ("input", "run", "cluster", "advanced")):
        return True
    # `project` alone is in both layouts; its new-only keys settle it.
    project = raw.get("project") or {}
    return isinstance(project, dict) and "defaults" in project


def _check_keys(block, schema, where):
    if block is None:
        return
    if not isinstance(block, dict):
        raise LayoutError(f"{where or 'the config'} must be a block of "
                          f"settings, got {block!r}.")
    for key, val in block.items():
        path = f"{where}.{key}" if where else key
        if key not in schema:
            if not where and key in _LEGACY_HINT:
                raise LayoutError(
                    f"'{key}' belongs to the old config layout. In this "
                    f"layout it is {_LEGACY_HINT[key]}.")
            close = difflib.get_close_matches(key, list(schema), n=1)
            hint = f" Did you mean '{close[0]}'?" if close else ""
            raise LayoutError(f"Unknown setting '{path}'.{hint} Known "
                              f"here: {', '.join(schema)}.")
        if isinstance(schema[key], dict):
            _check_keys(val, schema[key], path)


# ------------------------------
# 3. HELPERS
# ------------------------------
def _get(raw, path):
    """Value at a dotted path, or None when any level is missing."""
    node = raw
    for part in path.split("."):
        if not isinstance(node, dict) or node.get(part) is None:
            return None
        node = node[part]
    return node


def _put(out, path, value):
    node = out
    parts = path.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = copy.deepcopy(value)


def _pair(value, where):
    """A [first, second] list, either entry allowed to be null."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise LayoutError(f"{where} must be a list of two values, e.g. "
                          f"[2, 16]; got {value!r}.")
    return value[0], value[1]


# ------------------------------
# 4. TRANSLATION
# ------------------------------
def translate(raw):
    """New-layout dict -> internal settings dict (still merged with the
    defaults afterwards). Also returns notes for the log."""
    _check_keys(raw, SCHEMA, "")
    notes = []
    out = {}

    name = _get(raw, "project.name")
    for new_path, internal in _DIRECT:
        val = _get(raw, new_path)
        if val is not None:
            if isinstance(val, str) and name:
                val = val.replace("{name}", name)
            _put(out, internal, val)
    for new_path, key in _BOTH_DOMAINS.items():
        val = _get(raw, new_path)
        if val is not None:
            _put(out, f"domains.child.{key}", val)
            _put(out, f"domains.parent.{key}", val)

    # -- paths inside the two data sections ---------------------------------
    for section in ("raw_data", "user_data"):
        d = _get(out, f"{section}.dir")
        if isinstance(d, str) and name:
            _put(out, f"{section}.dir", d.replace("{name}", name))

    # -- project.defaults ----------------------------------------------------
    # false = no site defaults, a path = that file, null = the standard one.
    defaults = (raw.get("project") or {}).get("defaults")
    if defaults is False:
        _put(out, "project.no_defaults", True)
    elif isinstance(defaults, str):
        _put(out, "project.defaults_file", defaults)
    elif defaults is not None and defaults is not True:
        raise LayoutError(f"project.defaults must be a path or false, got "
                          f"{defaults!r}.")

    # -- cluster.nodes / cluster.cores ---------------------------------------
    nodes = _get(raw, "cluster.nodes")
    if nodes is not None:
        lo, hi = _pair(nodes, "cluster.nodes")
        if lo is not None:
            _put(out, "templates.values.min_nodes", lo)
        if hi is not None:
            _put(out, "templates.values.max_nodes", hi)
    cores = _get(raw, "cluster.cores")
    if cores is not None:
        parent, child = _pair(cores, "cluster.cores")
        if parent is not None:
            _put(out, "templates.values.npes_parent", parent)
        if child is not None:
            _put(out, "templates.values.npes_child", child)

    # -- which stages run ----------------------------------------------------
    # A stage is switched off by `enabled: false` on its own block; the rest
    # of the stage list stays whatever the defaults say.
    off = [stage for stage in ("templates", "report")
           if (raw.get(stage) or {}).get("enabled") is False]
    if off and _get(raw, "advanced.stages") is None:
        notes.append("disabled by enabled: false - "
                     + ", ".join(sorted(off)))
        out["_stages_off"] = off

    if _get(raw, "input.user_data.domain") is None and \
            _get(raw, "input.user_data.domain_from") is None and \
            (raw.get("input") or {}).get("user_data") is not None:
        notes.append("input.user_data: neither domain nor domain_from is "
                     "set - the site defaults decide the area of interest.")

    return out, notes


# ------------------------------
# 5. MESSAGES IN THE USER'S VOCABULARY
# ------------------------------
MESSAGE_NAMES = {
    "crs.aligned": "input.crs",
    "crs.domain_output": "input.domain_crs",
    "user_data.dir": "input.user_data.dir",
    "user_data.domain_from": "input.user_data.domain_from",
    "user_data.domain": "input.user_data.domain",
    "raw_data.dir": "input.raw_data.dir",
    "raw_data.aux_files": "input.aux_files",
    "templates.values.origin_time": "run.origin_time",
    "templates.values.length": "run.length",
    "templates.values.wrf_date": "run.wrf_date",
    "templates.values.nz_child": "domains.child.nz",
    "templates.values.nz_parent": "domains.parent.nz",
    "templates.values.node_cpus": "cluster.node_cpus",
    "templates.values.walltime": "cluster.walltime",
    "templates.values.hpc_user": "cluster.user",
    "templates.values": "run / cluster",
    "templates.topology_select": "advanced.cpu_topology.select",
    "templates.list_all_topologies": "advanced.cpu_topology.list_all",
    "templates.dir": "advanced.templates_dir",
    "domains.child.optimize_topology": "advanced.cpu_topology.optimize_child",
    "domains.parent.optimize_topology":
        "advanced.cpu_topology.optimize_parent",
    "domains.child.confirm_optimized":
        "advanced.cpu_topology.confirm_optimized",
    "domains.strict_nesting": "advanced.grid.strict_nesting",
    "domains.align_child_to_parent": "advanced.grid.align_child_to_parent",
    "clip.": "advanced.clip.",
    "merge.": "advanced.merge.",
    "boundary_cleanup": "advanced.boundary_cleanup",
    "stages": "advanced.stages",
}

_MESSAGE_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(k) for k in
                           sorted(MESSAGE_NAMES, key=len, reverse=True))
    + r")(?!\w)")


def to_new_names(text):
    """Rewrite internal setting names in a message into the new layout."""
    return _MESSAGE_RE.sub(lambda m: MESSAGE_NAMES[m.group(1)], text)
