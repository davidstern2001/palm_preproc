# Changelog

## 1.1.0

**Correctness**

- `boundary`: walls are now matched to boundary-truncated roofs by **geometry**
  (a wall lying under a removed roof's footprint) instead of by a shared
  building-id column. Id matching could delete walls of interior buildings
  whose id collided numerically with a removed roof's id.
- `boundary`: landcover under removed buildings is relabeled **per building** —
  local overlap threshold and a local majority vote from a ring around that
  building. Previously all removed roofs were dissolved into one footprint,
  which suppressed most relabels and assigned one global majority class to
  every parcel.
- `templates`: child `nz` is derived from the **mean absolute rooftop height**
  (terrain + building, referenced to the domain's lowest terrain point — the
  static driver's datum) rather than the relative building height, which
  under-sized `nz` on sloping terrain.
- `templates`: `origin_time` is written **unquoted** in pmeteo configs;
  palm_meteo parses it as a YAML timestamp and rejected the quoted form.
- `templates`: the "values not set" warning no longer reports `nz` as missing
  when it was successfully auto-calculated.

**Features**

- The pgem PostgreSQL password is read from the `PALM_PGEM_PASSWORD` environment variable instead of being stored in the templates; if unset, the `<pg_password>` placeholder is left for manual filling.

- Vector inputs may be any format GeoPandas reads (Shapefile, GeoPackage, GeoJSON, ...); the clipped vector outputs are always written as Shapefiles for PALM-GeM, converted after clipping.

- Seasonal, month-dependent temperatures in `defaults/templates/seasonal.yaml`,
  selected by the month of `origin_time`: water body temperatures per landcover
  type (`water_pars_temp`), soil temperature/moisture, and arbitrary scalar p3d
  parameters. Values are in Kelvin and substituted verbatim.
- Cyclic-boundary p3d templates (`template_cyclic_*`), enabled with
  `templates.cyclic: true`. These are a starting skeleton and log a TODO
  warning when used.
- `templates.values.hpc_user` and `wrf_dir` fill `<hpc_user>` / `<wrf_dir>` in
  the pgem and pmeteo templates, so each user sets their cluster identity once
  instead of editing templates.
- `templates.values.length` accepts a duration string with a unit
  (`"29 h"`, `"1 d"`) as well as a bare number of hours.
- `test.py`: builds a synthetic dataset in a temporary directory, runs the
  full pipeline and checks the outputs. Run it to verify a new environment.
  It defines its CRS explicitly rather than by EPSG code, so it does not
  depend on a healthy PROJ database, and reports a broken one as a warning
  about `crs.domain_output` rather than failing.

**Usability**

- Version and, in batch jobs, a "prompts auto-accept the recommended option"
  notice are logged at startup.
- Auto-selected processor topology and auto-accepted child sizing are now
  logged at INFO with what was chosen and what the alternatives were.
- Seasonal keys renamed to canonical PALM/PALM-GeM names
  (`water_pars_temp`, `soil_temperature`, `deep_soil_temperature`,
  `p3d_parameters`); the previous `*_k` names still work with a deprecation
  warning.
- README rewritten: table of contents, data-directory tree, per-stage sections.

## 1.0.0

Initial pipeline: `domains → clip → merge → boundary → templates → report`,
resumable state, topology-aware domain sizing, layered configuration.
