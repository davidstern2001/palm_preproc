"""YAML config loading + defaults for palm_preproc."""

import copy
from pathlib import Path

import yaml

from .log import get_logger

log = get_logger()

# ------------------------------
# 1. LAYER REGISTRY
# ------------------------------
# Raster layers are always taken from DATA_raw (never merged with user input).
RASTER_LAYERS = ("dem", "buildings")
# Vector layers may be overridden by user (child-domain) input via the merge stage.
VECTOR_LAYERS = ("landcover", "roofs", "walls", "trees")
ALL_LAYERS = RASTER_LAYERS + VECTOR_LAYERS

STAGES = ("domains", "clip", "merge", "boundary", "templates", "report")

# ------------------------------
# 2. DEFAULTS
# ------------------------------
DEFAULTS = {
    "project": {
        "name": "palm_preproc",
        # Base directory for ALL relative paths in this config. Default (null)
        # is the directory of the YAML file itself; set this when configs are
        # stored centrally (e.g. in config/) away from the project data.
        "root": None,
        "output_dir": ".",
        "state_file": None,          # default: <output_dir>/palm_preproc_state.json
        # true: regenerate ALL outputs, ignoring resume state and replacing
        # existing generated templates and aux files (import_process.log
        # included - use with care).
        "overwrite": False,
        # Site-wide defaults (defaults/default.yaml) are applied
        # automatically before this file. Point elsewhere with defaults_file,
        # or opt out entirely with no_defaults: true.
        "defaults_file": None,
        "no_defaults": False,
    },
    "crs": {
        # Axis-aligned CRS in which ALL computation and ALL data outputs live.
        "aligned": "EPSG:32633",
        # Optional different CRS for the two domain_*.shp rectangles only
        # (e.g. "EPSG:5514" for viewing over Czech data). Data layers always
        # stay in crs.aligned because they must match the PALM grid.
        "domain_output": None,
    },
    # The user's own (child-domain) data: same interface as raw_data
    # (dir + layers mapping of layer name -> filename). A layer participates
    # when its file exists; `false` disables it; a path outside `dir` is
    # allowed too. `domain` is required.
    "user_data": {
        "dir": "DATA_user",
        # Area of interest. Either an explicit domain.shp, or - when domain
        # is null - derive it from the full extent (bounding box) of one of
        # the user layers below via `domain_from` (e.g. "landcover"). An
        # explicit domain always wins when present.
        "domain": "domain.shp",
        "domain_from": None,
        "layers": {
            "landcover": "landcover.shp",
            "roofs": "roofs.shp",
            "walls": "walls.shp",
            "trees": "trees.shp",
        },
    },
    "raw_data": {
        "dir": "DATA_raw",
        "layers": {
            "dem": "dem.tif",
            "buildings": "buildings.tif",
            "landcover": "landcover.shp",
            "roofs": "roofs.shp",
            "walls": "walls.shp",
            "trees": "trees.shp",
        },
        # Auxiliary files placed into every generated data directory
        # (DATA_child, DATA_parent). Modes: "copy" = copy the file of the
        # same name from raw_data.dir; "create_empty" (alias "touch") =
        # create an empty file. Existing destinations are NEVER overwritten.
        "aux_files": {
            "surface_params.csv": "copy",
            "import_process.log": "create_empty",
        },
    },
    "domains": {
        "child": {
            "grid_size": 1.0,
            "buffer": 0.0,
            "min_power": 3,
            "optimize_topology": False,
            "topology_max_overhead": 0.10,
            "topology_opt": True,
            # Ask before accepting a topology-optimized (grown) child domain
            # instead of the baseline snap (interactive sessions only).
            "confirm_optimized": True,
        },
        "parent": {
            "grid_size": 2.0,
            # PALM convention: 400-500 m parent buffer around the child keeps
            # boundary effects (inflow adjustment, outflow damping) away from
            # the area of interest.
            "buffer": 500.0,
            "min_power": 3,
            "optimize_topology": True,
            "topology_max_overhead": 0.10,
            "topology_opt": True,
        },
        # Snap the child origin/extent onto the parent grid so PALM nesting
        # constraints (child edges on parent grid lines) hold by construction.
        "align_child_to_parent": True,
        # Fail loudly (abort the run) if the child origin offset or extent is
        # not EXACTLY on the parent lattice. Set to false only for
        # deliberately non-nested setups.
        "strict_nesting": True,
    },
    "clip": {
        # Resample clipped rasters onto the EXACT PALM grid of each domain
        # (origin, grid_size, width_pts x height_pts). Strongly recommended.
        "snap_rasters_to_grid": True,
        # Mask the buildings raster to this (final, possibly merged) vector
        # layer: cells outside its polygons become nodata. null disables.
        "buildings_mask_layer": "roofs",
        "buildings_nodata": -9999.0,
        "resampling": {"dem": "bilinear", "buildings": "nearest", "default": "nearest"},
        "workers": 4,
    },
    "report": {
        "file": None,                # default: <output_dir>/domains_report.txt
        "parent_name": "parent",
        "child_name": "child",
    },
    "templates": {
        "dir": None,                 # null -> defaults/templates/
        "nested": True,              # nested (parent+child) or single-domain set
        "single_domain": "parent",   # domain used when nested: false
        "case": None,                # default: project.name
        "overwrite": False,          # existing generated files are kept
        # true: present Hynek Reznicek's valid topologies and let the user
        # pick (falls back to the recommendation when non-interactive);
        # false: always take the recommendation silently.
        "topology_select": True,
        "values": {
            "origin_time": None,     # UTC, e.g. "2023-08-23 17:00:00"
            "nz_parent": None,
            "nz_child": None,
            "length": None,          # simulation length [h]: end_time (p3d)
                                     # and pmeteo length (was simulation_hours)
            "wrf_date": None,        # WRF path date, e.g. "2023-08-23"
            "npes_parent": None,     # TARGET core counts: steer the topology
            "npes_child": None,      #   recommendation (null -> most cores)
            "node_cpus": None,       # cpus per node (-T): enables the nodes
                                     #   column and the node limits below
            "min_nodes": None,       # hide topologies below this node count
            "max_nodes": None,       # hide topologies above this node count
            "npex_parent": None,     # null -> topology chooser (see topology_select)
            "npey_parent": None,
            "npex_child": None,
            "npey_child": None,
        },
    },
    "boundary_cleanup": {
        "enabled": True,
        "domains": ["child", "parent"],
        # Buffer ring (metres) used to find "surrounding" landcover when
        # relabeling a deleted building's footprint.
        "buffer": 20.0,
        "landcover_columns": ["code", "type"],
        "roofs_id_column": None,   # null -> auto-detect (bid/rid/lid/id)
        "walls_id_column": None,
    },
    "merge": {
        # "difference": raw geometry outside the priority footprint is kept and
        #                CUT along the footprint boundary (no gaps, no overlaps).
        # "predicate":  keep whole features by spatial predicate (original
        #                merge.py behaviour).
        "mode": "difference",
        "predicate": "within",       # used only in mode: predicate
        "fix_geometry": True,        # buffer(0) on polygon layers after merge
        "int_columns": ["lid", "type", "code"],   # cast to nullable Int64
        # Regenerate a unique 1..N id column after merging, per layer
        # (avoids id collisions between user and raw features).
        "reassign_id": {"landcover": "lid"},
        # Provenance column added to every merged layer: 1 = user feature,
        # 0 = raw feature (created if absent; null disables).
        "mark_user_column": "user",
        # How raw is removed where user data exists, per layer:
        #  "coverage": cut raw exactly along the user geometry, so raw fills
        #     the true gaps between user features (right for wall-to-wall
        #     layers like landcover).
        #  "footprint": drop ALL raw inside the user-domain footprint, so
        #     inside the domain the layer is user-only and raw appears only
        #     outside (right for buildings/roofs/walls/trees, where gaps
        #     between features are streets that must stay empty).
        # Unlisted layers default to "footprint".
        "raw_fill": {
            "landcover": "coverage",
            "roofs": "footprint",
            "walls": "footprint",
            "trees": "footprint",
        },
        # Optional per-layer mask polygon (a path to a .shp) defining that
        # layer's OWN keep-user boundary: user features inside the mask are
        # kept, raw fills outside it - overriding the shared domain footprint
        # for that layer only. E.g. {trees: mask_trees.shp}. Paths resolve
        # against user_data.dir (bare filename) or project.root (a path).
        "masks": {},
        # "domain": user priority over the whole domain.shp footprint;
        # "user_coverage": raw data fills holes inside the user coverage
        # (polygon layers only; point/line layers fall back to "domain").
        "priority_source": "domain",
        "workers": 4,
    },
    "stages": list(STAGES),
}


# ------------------------------
# 3. HELPERS
# ------------------------------
def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class ConfigError(Exception):
    pass


# ------------------------------
# 4. CONFIG OBJECT
# ------------------------------
class Config:
    """Resolved configuration. Paths are absolute (relative to the YAML file).

    Three layers are merged, most specific last: hardcoded DEFAULTS -> the
    site-wide defaults/default.yaml (auto-loaded unless disabled) ->
    this project's own YAML.
    """

    # Sibling of template.yaml at the repo root, i.e. .../palm_preproc/defaults
    DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "defaults"

    def __init__(self, yaml_path):
        self.yaml_path = Path(yaml_path).resolve()
        raw = yaml.safe_load(self.yaml_path.read_text()) or {}

        house = {}
        house_path = self._house_defaults_path(raw)
        if house_path and house_path.exists():
            house = yaml.safe_load(house_path.read_text()) or {}
            log.debug(f"Config: applying site defaults from {house_path}")
        elif house_path:
            log.debug(f"Config: no site defaults file at {house_path}")

        self.d = _deep_merge(DEFAULTS, house)
        self.d = _deep_merge(self.d, raw)
        self._resolve_paths()
        self._validate()

    def _house_defaults_path(self, raw):
        pcfg = raw.get("project") or {}
        if pcfg.get("no_defaults"):
            return None
        override = pcfg.get("defaults_file")
        if override:
            p = Path(override)
            return p if p.is_absolute() else (self.yaml_path.parent / p).resolve()
        return self.DEFAULT_CONFIG_DIR / "default.yaml"

    def _resolve(self, p):
        p = Path(p)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    def _resolve_paths(self):
        d = self.d
        root = d["project"].get("root")
        if root:
            root = Path(root)
            self.base_dir = (root if root.is_absolute()
                             else (self.yaml_path.parent / root).resolve())
        else:
            self.base_dir = self.yaml_path.parent
        d["project"]["root"] = str(self.base_dir)
        d["project"]["output_dir"] = self._resolve(d["project"]["output_dir"])
        # state_file: null -> default path; false -> disabled (no state file);
        # anything else -> an explicit path. A bare `true` is meaningless and
        # is treated as null, with a warning: YAML turns unquoted yes/no/on/
        # off into booleans, so this is an easy thing to write by accident.
        sf = d["project"]["state_file"]
        if sf is True:
            log.warning("Config: project.state_file: true is not a path; "
                        "using the default. Use `null` for the default, "
                        "`false` to disable, or quote an explicit path.")
            sf = None
        if sf is None:
            d["project"]["state_file"] = d["project"]["output_dir"] / "palm_preproc_state.json"
        elif sf is False:
            d["project"]["state_file"] = False
        else:
            d["project"]["state_file"] = self._resolve(sf)
        # Back-compat: map the old inputs.{user_data_dir,domain,user_layers}
        # keys onto the unified user_data section.
        legacy = d.pop("inputs", None)
        if legacy:
            log.warning("Config: the 'inputs:' section is deprecated; use "
                        "'user_data:' (dir/domain/layers, like raw_data).")
            if legacy.get("user_data_dir"):
                d["user_data"]["dir"] = legacy["user_data_dir"]
            if legacy.get("domain"):
                d["user_data"]["domain"] = legacy["domain"]
            for k, v in (legacy.get("user_layers") or {}).items():
                d["user_data"]["layers"][k] = v

        def in_dir(base, value):
            """Resolve a filename against `base`, a path against the root."""
            p = Path(value)
            if p.is_absolute() or len(p.parts) > 1:
                return self._resolve(value)
            return base / value

        d["user_data"]["dir"] = self._resolve(d["user_data"]["dir"])
        user_dir = d["user_data"]["dir"]
        if d["user_data"]["domain"]:
            d["user_data"]["domain"] = in_dir(user_dir, d["user_data"]["domain"])
        else:
            d["user_data"]["domain"] = None
        # A user layer participates when its resolved file exists; `false`
        # disables it explicitly.
        d["user_data"]["layers"] = {
            k: in_dir(user_dir, v)
            for k, v in (d["user_data"]["layers"] or {}).items()
            if v and in_dir(user_dir, v).exists()
        }
        d["raw_data"]["dir"] = self._resolve(d["raw_data"]["dir"])
        d["merge"]["masks"] = {
            layer: in_dir(user_dir, p)
            for layer, p in (d["merge"].get("masks") or {}).items()
            if p
        }
        if d["report"]["file"]:
            d["report"]["file"] = self._resolve(d["report"]["file"])
        if d["templates"]["dir"]:
            d["templates"]["dir"] = self._resolve(d["templates"]["dir"])

    def _validate(self):
        d = self.d
        dfrom = d["user_data"].get("domain_from")
        if d["user_data"]["domain"]:
            if not Path(d["user_data"]["domain"]).exists():
                raise ConfigError(
                    f"Domain shapefile not found: {d['user_data']['domain']} "
                    f"(place domain.shp in user_data.dir, set "
                    f"user_data.domain explicitly, or use user_data.domain_from)."
                )
        elif dfrom:
            if dfrom not in d["user_data"]["layers"]:
                raise ConfigError(
                    f"user_data.domain_from = '{dfrom}' but that layer is not "
                    f"present in user_data.layers (found: "
                    f"{sorted(d['user_data']['layers'])})."
                )
        else:
            raise ConfigError(
                "No area of interest: set user_data.domain (a domain.shp) or "
                "user_data.domain_from (a user layer to take the extent of)."
            )
        # Vector layers and the buildings raster may be user-supplied; dem
        # always comes from raw_data (no user-DEM merge is defined).
        user_mergeable = VECTOR_LAYERS + ("buildings",)
        for name in d["user_data"]["layers"]:
            if name not in user_mergeable:
                raise ConfigError(
                    f"user_data.layers.{name}: only {user_mergeable} can be "
                    f"user-supplied (dem always comes from raw_data)."
                )
        for name in ALL_LAYERS:
            if name not in d["raw_data"]["layers"]:
                raise ConfigError(f"raw_data.layers.{name} missing from config.")
        for layer, p in d["merge"].get("masks", {}).items():
            if layer not in ALL_LAYERS:
                raise ConfigError(f"merge.masks.{layer}: unknown layer "
                                  f"(valid: {ALL_LAYERS}).")
            if not Path(p).exists():
                raise ConfigError(f"merge.masks.{layer} not found: {p}")
        unknown = set(d["stages"]) - set(STAGES)
        if unknown:
            raise ConfigError(f"Unknown stage(s): {sorted(unknown)}; valid: {STAGES}")
        ratio = d["domains"]["parent"]["grid_size"] / d["domains"]["child"]["grid_size"]
        if d["domains"]["align_child_to_parent"] and abs(ratio - round(ratio)) > 1e-9:
            log.warning(
                f"parent/child grid ratio {ratio:g} is not an integer; PALM "
                f"nesting normally requires an integer ratio. Child-to-parent "
                f"alignment will be skipped."
            )

    # -- convenience accessors -------------------------------------------
    def __getitem__(self, key):
        return self.d[key]

    @property
    def output_dir(self):
        return self.d["project"]["output_dir"]

    def raw_layer_path(self, layer):
        return self.d["raw_data"]["dir"] / self.d["raw_data"]["layers"][layer]

    def user_layer_path(self, layer):
        return self.d["user_data"]["layers"].get(layer)

    def domain_source(self):
        """(path, kind) of the area-of-interest source: an explicit
        domain.shp ('domain') or a user layer to take the extent of
        ('extent')."""
        if self.d["user_data"]["domain"]:
            return self.d["user_data"]["domain"], "domain"
        layer = self.d["user_data"]["domain_from"]
        return self.d["user_data"]["layers"][layer], "extent"

    def output_layer_filename(self, layer):
        """Filename a layer is WRITTEN under in DATA_<domain>. Vector inputs
        may be any format GeoPandas reads (.shp, .gpkg, .geojson, ...), but
        the outputs are always Shapefiles, because PALM-GeM consumes
        Shapefiles. Converting happens after clipping, when the data has been
        cut to the (much smaller) domain. Raster layers keep their filename.
        """
        fname = self.d["raw_data"]["layers"][layer]
        if layer in VECTOR_LAYERS:
            return str(Path(fname).with_suffix(".shp"))
        return fname

    def data_dir(self, domain_name):
        return self.output_dir / f"DATA_{domain_name}"

    def as_plain_dict(self):
        """JSON-serialisable copy for hashing. The `report` section is
        excluded so tuning npes/names never invalidates the resume state,
        and so is `templates` (regenerated on demand, never overwriting)."""
        import json
        d = json.loads(json.dumps(self.d, default=str))
        d.pop("report", None)
        d.pop("templates", None)
        return yaml.safe_load(yaml.safe_dump(d))
