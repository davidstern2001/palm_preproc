# palm_preproc

**Preprocessing pipeline for PALM static-driver input data** — companion to `palm_postproc`.

From a single area-of-interest shapefile, `palm_preproc` builds nested child/parent domain rectangles snapped to PALM-friendly sizes, clips a whole-city raw dataset to each domain on the exact PALM grid, optionally merges in your own higher-quality layers, cleans up buildings sliced by the domain edge, and writes ready-to-edit PALM / palm_meteo / PALM-GeM run-configuration files with a processor topology chosen for efficient use of the cluster.

---

## Table of contents

- [How it works (the six stages)](#how-it-works-the-six-stages)
- [Installing and running](#installing-and-running)
- [Data you provide, files you get back](#data-you-provide-files-you-get-back)
- [Configuration](#configuration)
  - [Site-wide defaults vs. per-project config](#site-wide-defaults-vs-per-project-config)
  - [Key options](#key-options)
- [Stage details](#stage-details)
  - [domains — sizing and nesting](#domains--sizing-and-nesting)
  - [clip — onto the PALM grid](#clip--onto-the-palm-grid)
  - [merge — your data over the raw data](#merge--your-data-over-the-raw-data)
  - [boundary — cleaning up sliced buildings](#boundary--cleaning-up-sliced-buildings)
  - [templates — namelists, nz, and the topology chooser](#templates--namelists-nz-and-the-topology-chooser)
  - [report — the &nesting_parameters block](#report--the-nesting_parameters-block)
- [Command-line reference](#command-line-reference)
- [Standalone tools](#standalone-tools)
- [Things worth knowing](#things-worth-knowing)
- [Authors](#authors)

---

## How it works (the six stages)

The pipeline runs six stages in order. Each stage is **resumable** — its result is recorded in a JSON state file and only recomputed when the relevant config changes — and the `clip` and `merge` stages process layers in parallel.

| # | Stage | What it produces |
|---|-------|------------------|
| 1 | `domains` | Child (no buffer) and parent (buffered) rectangles, snapped to PALM-friendly sizes, topology-aware, with the child aligned exactly on the parent lattice |
| 2 | `clip` | Every raw layer clipped to both domains; rasters resampled onto each domain's exact PALM grid |
| 3 | `merge` | Your own child-domain layers coalesced over the clipped raw layers, with your data taking priority inside the area of interest |
| 4 | `boundary` | Buildings truncated at the domain edge removed (roofs + their walls), and the landcover under them relabeled |
| 5 | `templates` | PALM `_p3d`, palm_meteo, and PALM-GeM configs filled in — interactive topology choice, auto-calculated `nz`, restart namelists |
| 6 | `report` | `domains_report.txt`: case summary, nesting validation, and the `&nesting_parameters` block |

The default order lives in `defaults/default.yaml` as `stages: [domains, clip, merge, boundary, templates, report]`.

---

## Installing and running

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Check that the environment works. This builds a small synthetic dataset
#    in a temporary directory, runs the whole pipeline over it and verifies
#    the outputs — no real data needed, a few seconds.
python test.py

# 3. Make a config from the annotated template, then edit it
cp template.yaml config/my_project.yaml
#    → set at least project.name and project.root

# 4. Run
python run_preproc.py -c config/my_project.yaml
```

If `test.py` fails, the message says whether a dependency is missing or a
stage broke — fix that before pointing the pipeline at real data.

It may also report that **EPSG code lookups fail** while still finishing
successfully. That means the environment's PROJ database (`proj.db`) is stale
or mismatched — commonly a leftover `PROJ_DATA`/`PROJ_LIB` variable, or conda
and pip both supplying GDAL/PROJ. The pipeline itself is unaffected, because it
reads the CRS embedded in your data files rather than resolving EPSG codes; only
`crs.domain_output` reprojection needs the database and would fail. Fix it by
clearing the stale variable or reinstalling GDAL/PROJ from a single source.

Useful variations:

```bash
# Validate the config and print the plan without writing anything
python run_preproc.py -c config/my_project.yaml --dry-run

# Ignore the resume state and rebuild everything
python run_preproc.py -c config/my_project.yaml --force

# Re-run only certain stages (e.g. after tweaking report labels)
python run_preproc.py -c config/my_project.yaml --stages report
```

Before running, make sure the data is in place (paths are set in the config):

```
DATA_raw/    dem.tif  buildings.tif  landcover.shp  roofs.shp  walls.shp  trees.shp
DATA_user/   (optional) your own landcover/roofs/walls/trees.shp and/or buildings.tif
             — plus, if you don't derive it, an explicit domain.shp
```

---

## Data you provide, files you get back

**Input formats.** Vector layers may be in any format GeoPandas/GDAL reads — Shapefile, GeoPackage (`.gpkg`), GeoJSON, etc. — set per layer in `raw_data.layers` / `user_data.layers` (e.g. `landcover: landcover.gpkg`). The pipeline always **writes** the clipped vector outputs as Shapefiles, because PALM-GeM consumes Shapefiles; the conversion happens after clipping, on the much smaller domain-sized data. GeoPackage inputs must be single-layer.

**Inputs.** `DATA_raw` is the permanent whole-city source dataset — it is only ever read, never modified. `DATA_user` holds anything project-specific: the area of interest (either an explicit `domain.shp`, or derived from the extent of one of your layers via `user_data.domain_from`), plus any higher-quality layers you want to override the raw data with inside that area. A user layer participates simply by existing; set it to `false` to ignore a file that's present, or to a path to use a file elsewhere.

**The layout at a glance.** Everything hangs off `project.root`; `DATA_raw`
is a sibling of the per-project directories, shared by all of them:

```
<project.root>/                        e.g. /home/<you>/palm/DATA
├── DATA_raw/                          whole-city source data - only ever READ
│   ├── dem.tif
│   ├── buildings.tif
│   ├── landcover.shp                  (each .shp with its .dbf/.shx/.prj sidecars)
│   ├── roofs.shp
│   ├── walls.shp
│   ├── trees.shp
│   └── surface_params.csv             aux file, copied into each domain dir
│
└── <project>/                         one directory per project (= output_dir)
    ├── DATA_user/                     your input for this project
    │   ├── domain.shp                 optional - or derived from a layer
    │   │                              via user_data.domain_from
    │   ├── landcover.shp              optional user layers: present = used,
    │   ├── roofs.shp                  absent = raw data only
    │   ├── walls.shp
    │   ├── trees.shp
    │   ├── buildings.tif
    │   └── mask_trees.shp             optional per-layer merge mask
    │                                  (merge.masks)
    │
    │   # everything below is GENERATED by the pipeline:
    ├── domain_child.shp
    ├── domain_parent.shp
    ├── DATA_child/                    final per-domain data layers
    │   ├── dem.tif, buildings.tif, landcover.shp, roofs.shp, ...
    │   ├── _clipped/                  pre-merge raw versions, for inspection
    │   ├── surface_params.csv
    │   └── import_process.log
    ├── DATA_parent/                   same structure as DATA_child/
    ├── <case>_p3d, <case>_p3d_N02
    ├── <case>_p3dr, <case>_p3dr_N02
    ├── pgem_<case>.yaml, pgem_<case>_N02.yaml
    ├── pmeteo_<case>.yaml, pmeteo_<case>_N02.yaml
    ├── domains_report.txt
    └── palm_preproc_state.json
```

Starting a new project therefore only means: make `<project>/DATA_user/`, put
your layers in it, copy `template.yaml` to `config/<project>.yaml`, point
`project.root` at the tree above, and run.

**Outputs.** Everything is written under `project.output_dir` (relative to `project.root`). For a case named `<case>` (defaults to `project.name`):

| File(s) | Stage | Purpose |
|---------|-------|---------|
| `domain_child.shp`, `domain_parent.shp` | domains | Snapped rectangles; `origin_x/y`, `nx/ny`, and extents are in the attribute table |
| `DATA_child/`, `DATA_parent/` | clip + merge | The clipped (and merged) data layers, plus `surface_params.csv` and `import_process.log` |
| `<case>_p3d`, `<case>_p3d_N02` | templates | PALM namelists (parent + child) with grid, topology, and time values filled in |
| `<case>_p3dr`, `<case>_p3dr_N02` | templates | Restart namelists — identical, but `initializing_actions` is switched to reading restart data |
| `pgem_<case>.yaml`, `pgem_<case>_N02.yaml` | templates | Static-driver generator configs (dx/dy/dz, nx/ny, cent_x/cent_y, origin_time) |
| `pmeteo_<case>.yaml`, `pmeteo_<case>_N02.yaml` | templates | palm_meteo configs (dz, nz, origin_time, length, WRF paths via `wrf_date`) |
| `domains_report.txt` | report | Case summary, nesting validation, `&nesting_parameters`, and a summary table |
| `palm_preproc_state.json` | — | Resume state |

With `templates.nested: false` only the single-domain set is produced (`<case>_p3d`, `<case>_p3dr`, `pgem_<case>.yaml`, `pmeteo_<case>.yaml`), using the domain named in `templates.single_domain`.

---

## Configuration

### Site-wide defaults vs. per-project config

Configuration is layered, applied in this order:

```
DEFAULTS (hardcoded)  →  defaults/default.yaml  →  config/<project>.yaml
```

Later layers override earlier ones, and the merge is **deep** — you can override a single nested key (say `domains.parent.buffer`) without restating the rest of the block.

- **`defaults/default.yaml`** is the baseline for *every* project on this machine. Put things here that rarely change between projects: buffers, resampling methods, `raw_fill` rules, cluster facts (`node_cpus`, node limits), boundary-cleanup settings, `nz` auto-calc factors, the stage list. The run-configuration skeletons it points at live in `defaults/templates/`.
- **`config/<project>.yaml`** then only needs to state what's genuinely project-specific: name, root, output directory, your `user_data` layers, simulation time (`origin_time`, `length`, `wrf_date`), and any deliberate override of a default.

To skip the site defaults for one project use `project.no_defaults: true`; to point at a different defaults file use `project.defaults_file: <path>`. The fully-commented `template.yaml` documents every setting and works as a standalone config too.

### Key options

A minimal, realistic project config:

```yaml
project:
  name: my_project
  root: /home/stern/palm/DATA        # base for all relative paths below
  output_dir: ./my_project

user_data:
  dir: ./my_project/DATA_user
  domain: null                       # null → derive the area of interest…
  domain_from: landcover             #   …from this layer's full extent
  layers:
    buildings: buildings.tif
    landcover: landcover.shp
    roofs:     roofs.shp
    walls:     walls.shp
    trees:     trees.shp

domains:
  parent:
    buffer: 400.0                    # metres around the final child (PALM convention: 400–500 m)

templates:
  values:
    origin_time: "2023-08-23 19:00:00"   # UTC
    length: 29                            # simulation length [h]
    wrf_date: "2023-08-23"                # WRF path date
```

Because defaults are inherited deeply, a project config really can be this short — every other setting comes from `defaults/default.yaml`.

---

## Stage details

### domains — sizing and nesting

Given the area of interest, the child rectangle is built with no buffer and the parent with a buffer around the *final* child (default 400–500 m, enough to keep boundary effects away from the region of interest). Both are snapped up to PALM-friendly sizes.

With `optimize_topology: true`, sizing is **topology-aware**: rather than just snapping up, the stage searches a small window (up to `topology_max_overhead`, default +10 % per axis) for the grid size that admits the most valid processor decompositions — a "joint score" equal to the number of valid `(npex, npey)` layouts that size allows. Preferring a high-scoring size here means the topology chooser in the `templates` stage has many good options later. When optimization grows the *child* beyond the plain snap, you're asked to confirm interactively (`confirm_optimized: true`); in batch runs it proceeds automatically.

`align_child_to_parent: true` places the child origin and extent exactly on the parent grid. If the result is off the parent lattice, `strict_nesting: true` **aborts** (see [Strict nesting](#things-worth-knowing)); set it to `false` to downgrade to a warning.

### clip — onto the PALM grid

Every raw layer is clipped to both domains. With `snap_rasters_to_grid: true`, rasters are resampled onto each domain's exact PALM grid (per-layer resampling methods: bilinear for the DEM, nearest for buildings, etc.). `buildings.tif` is masked to the final roofs of each domain (`buildings_mask_layer: roofs`), so heights survive only under roof polygons.

### merge — your data over the raw data

Where you supply a layer, it takes priority inside the area of interest and the clipped raw layer fills everything outside it, up to the domain rectangle. Two modes:

- **`difference`** (default) cuts raw geometry along the priority boundary, so the seam has neither gaps nor overlaps. Per-layer `raw_fill` controls how raw is removed: `coverage` (cut raw exactly along your geometry — right for wall-to-wall layers like landcover) or `footprint` (drop *all* raw inside the domain footprint — right for buildings/roofs/walls/trees, whose inter-feature gaps are streets that must stay empty).
- **`predicate`** keeps whole features by a spatial predicate (the original behaviour).

Post-merge cleanup: `fix_geometry` runs `buffer(0)` on polygon layers; `int_columns` casts attributes to nullable Int64; `reassign_id` regenerates unique 1..N ids so user and raw features never collide; `mark_user_column` records provenance (1 = user, 0 = raw). A per-layer **mask** (`merge.masks`, e.g. `trees: mask_trees.shp`) can give one layer its own keep-user boundary, different from the shared domain footprint.

### boundary — cleaning up sliced buildings

Clipping to a rectangle slices any building straddling the edge, which is physically wrong for PALM. Running on each domain's *final* layers (after clip and merge), this stage:

1. **Deletes roof polygons that touch the domain boundary** — they were truncated by clipping — instead of keeping partial slivers.
2. **Deletes those buildings' walls, matched by geometry.** A wall is removed only if it actually lies under a removed roof's footprint (representative-point-inside, plus a majority-area test for walls straddling the edge). Matching is geometric rather than by a shared id column, so walls of interior buildings are never removed by an id collision.
3. **Relabels the landcover under each removed building**, per building, to the area-weighted majority `code`/`type` in a ring around *that* building (`boundary_cleanup.buffer`, default 20 m). Each removed building is voted independently, so no orphaned "building" landcover is left and neighbouring removals don't pollute each other's vote.
4. **Re-masks `buildings.tif`** to the cleaned roofs, removing heights for the deleted buildings.

Runs for both domains by default; disable with `boundary_cleanup.enabled: false`.

### templates — namelists, nz, and the topology chooser

This stage fills the run-configuration skeletons in `defaults/templates/` for the case.

**Values you must supply** — `origin_time` (UTC), `length` (simulation hours), and `wrf_date` — come from `templates.values`; if unset, the `<>` placeholders stay and a warning names them.

**`nz` is calculated for you** unless you set `nz_parent`/`nz_child` explicitly:

- **Child** `nz` targets the roughness sublayer: `dz·nz = child_nz_height_factor × mean absolute rooftop height`. Each building cell's rooftop is `terrain + building_height`, referenced to the domain's *lowest* terrain point — the same datum PALM's static driver uses (`min(zt) → z = 0`). The mean is taken over building cells only, so a bare hill in the domain does not inflate `nz`, while buildings standing above the domain floor count at their true absolute height. This needs the DEM and buildings rasters on the same grid (guaranteed with `clip.snap_rasters_to_grid: true`); if they aren't aligned, or there's no DEM, it falls back to the plain above-terrain building height and says so in the log.
- **Parent** `nz` targets `parent_target_height_m` (~2000 m).

`nz` is then snapped up to a multiple of the domain's multigrid stride (`2^min_power`). Both factors live in `templates.values`, and an explicit `nz_parent`/`nz_child` overrides the calculation entirely.

**Choosing the processor topology.** The stage enumerates valid `(npex, npey)` layouts using Hynek Řezníček's rules — per-domain constraints, plus for nested runs the joint rule that the parent-to-child core ratio lie between 1:3 and 1:1 — and presents them as a numbered list (`topology_select: true`): pick a number, or press Enter for the recommended one (the capitalised default in `[Y/n]`-style prompts). By default the list shows only the best-packed option per whole-node count; `list_all_topologies: true` shows every one.

The recommendation favours **efficient node usage**: with `node_cpus` set, a job is billed `ceil(cores / node_cpus)` whole nodes, so a layout landing just *below* a whole-node boundary (e.g. 14.85 nodes) packs better than one just above (14.03) and is preferred; ties break toward more cores. Set `npes_parent`/`npes_child` to steer toward target core counts instead, or pin `npex_*`/`npey_*` to bypass the chooser. `min_nodes`/`max_nodes` hide layouts outside your node window. In non-interactive runs the recommendation is taken automatically. The chosen topology is saved in the state and is what the report documents, so the report always reflects the actual run.

Generated files are **never overwritten** on reruns unless you set `templates.overwrite: true`, and the `templates:` section is excluded from the resume-state hash.

### report — the &nesting_parameters block

`domains_report.txt` contains a short case description, nesting validation, a summary table, and a minimal namelist block with `domain_layouts` — all PALM strictly needs to define the nest geometry. Per domain:

| column | meaning |
|--------|---------|
| `name` | domain label (from `report.parent_name` / `report.child_name`) |
| `domain id` | unique id; the root/parent domain is `1` |
| `parent id` | id of the domain it nests into; `-1` marks the root |
| `npes` | MPI cores for this domain — the `npex × npey` product chosen in `templates` |
| `llx`, `lly` | offset [m] of the child SW corner from the parent SW corner; a non-negative integer multiple of the parent grid size (`0.0, 0.0` for the root) |

Two optional parameters you may add by hand: `nesting_mode` (`'one-way'`, the usual choice for urban microclimate, or `'two-way'`) and `nesting_datatransfer_mode` (`'cascade'`/`'mixed'`/`'overlap'`; `'mixed'` is a reasonable default).

The report also validates the geometry these values rely on: llx/lly on the parent grid, child containment, at least two parent cells of clearance on every side, and an integer grid-spacing ratio ≤ 5.

---

## Command-line reference

| Flag | Description |
|------|-------------|
| `-c CONFIG` | Path to the YAML config file (required) |
| `-n`, `--dry-run` | Validate and print the plan; write nothing |
| `--force` | Ignore the resume state and rerun everything |
| `--stages S [S ...]` | Run only these stages (e.g. `domains clip merge boundary templates report`) |
| `-v`, `--verbose` | DEBUG-level logging |
| `-q`, `--quiet` | WARNING-level logging only |
| `--log-datetime` | Prepend full date+time to log lines (default is HH:MM:SS) |

Logging follows the PALM-GeM style (Bureš & Resler, ICS CAS): timestamped lines, bold step announcements, indented detail, level tags only for warnings and errors.

---

## Standalone tools

Two stages double as standalone command-line tools, replacing the original scripts:

```bash
# gen_bounding_rectangle — build one snapped, optionally topology-optimized rectangle
python -m palm_preproc.steps.domains rozsah.shp domain.shp --buffer 400 --optimize-topology

# merge — coalesce a child-domain layer (priority inside) with a parent layer (outside)
python -m palm_preproc.steps.merge domain_child.shp trees.shp trees_parent.shp merged.shp
```

---

## Things worth knowing

- **CRS.** Data layers are always written in `crs.aligned` (default `EPSG:32633`) because they must match the PALM grid. `crs.domain_output` only reprojects the two `domain_*.shp` rectangles — after reprojection out of UTM these are rotated quadrilaterals, so never take `.total_bounds` of them downstream. Use the `origin_x`/`origin_y` attributes.
- **User `buildings.tif`.** The DEM always comes from `DATA_raw`, but the four vector layers *and* `buildings.tif` can be user-supplied. A user buildings raster is composited over the clipped raw buildings on each domain's exact grid: inside the footprint your heights replace raw (cells where your raster is nodata fall back to raw, so no holes), outside it raw is kept. The roofs mask then runs on the merged result.
- **Inspecting merges.** When a user layer exists, its clipped-raw counterpart is kept in `DATA_*/_clipped/` for comparison; the merged result gets the final name.
- **Report tuning is cheap.** The `report:` section is excluded from the resume-state hash, so adjusting npes targets or names and rerunning `--stages report` never invalidates the rest of the pipeline.
- **Interactive vs. batch runs.** Two stages can ask a question: confirming a
  topology-optimized child size, and choosing the processor topology. With no
  terminal attached (a queued job), both auto-accept the recommended option
  rather than hanging. The log says so at startup and records what was chosen,
  so a batch log always documents the decisions — the same config simply
  prompts when you run it on a login node. Pin `npex_*`/`npey_*` in
  `templates.values`, or set `templates.topology_select: false` /
  `domains.child.confirm_optimized: false`, to remove the questions entirely.
- **Strict nesting guarantee.** The child origin offset (llx/lly) and extent must lie *exactly* on the parent lattice. Any sub-grid displacement — which would make the child static-driver content sit at coordinates shifted from where PALM places the nest — aborts the run with a `NestingError` in the `domains` stage and again in `report` validation (the report is still written for inspection). Set `domains.strict_nesting: false` to downgrade to warnings for deliberately non-nested setups. If you build static drivers from these outputs with external tools, keep using the `origin_x`/`origin_y` attributes, or the guarantee is lost downstream.

---

## Authors

This code was written by [Claude.ai](https://claude.ai), configured and directed by **David Stern, Mgr.** (IPR Praha), based on his standalone scripts (`gen_bounding_rectangle`, `merge`, `comp_bounding_box`, `palm_comp_nesting_param`, `landcover_processing`).

The processor-topology rules and configuration enumeration (`steps/topology.py`, also used by topology-aware domain sizing) are by **Hynek Řezníček** (reznicek@cs.cas.cz), from his `procesor_topology.py` and `procesor_topology_nested.py`.
