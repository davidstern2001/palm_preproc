# Changelog

## 1.2.0

**Logging**

- The startup header is no longer printed twice. Attaching the default log
  file clears the handlers, so the header was re-emitted to reach the file
  - and appeared twice on the terminal. It now goes to the file handler
  alone via `emit_to_file_only()`.

**Correctness**

- `templates.values.length` accepts a bare number written as a STRING
  (`length: "29"`, which is also what a plain `29` becomes once quoted).
  It previously fell through the unit regex and returned None, so the
  simulation length was silently dropped and `<time>` /
  `<time_in_seconds>` were left as placeholders in the generated p3d.

- `_sub_keyed` anchors the key on its left, so a short key can no longer
  match the TAIL of a longer one. Without the anchor, substituting `nz`
  also fired inside `max_nz = <>` and `dz` inside `ref_dz = <>`, writing
  the wrong value into a namelist with no error anywhere. The shipped
  skeletons happen to contain no such pair, but adding one would have
  corrupted a run silently.

- The `domains` stage now really does reuse the geometry recorded in the
  resume state. The branch that did so was guarded by
  `state.is_done("domains") and ...`, which is unreachable — the function
  returns early on `is_done("domains")` a few lines above. In practice
  `--stages domains` and `--stages domains clip` fell through, recomputed
  the rectangles and re-prompted for the child size instead of reusing the
  decision already made. Keying on the saved specs alone makes the
  documented behaviour real; `--force` still recomputes.

- `boundary`: a landcover parcel claimed by several removed buildings now
  keeps the label from the FIRST building that claims it, which is what the
  docstring has always said. The previous code reassigned the parcel on
  every later hit, so the last building actually won.

**Performance**

- `boundary`: the per-building landcover lookups go through the layer's
  spatial index. Each removed building used to be tested against the whole
  landcover frame (two full `.intersection().area` passes), making the
  stage O(n_removed x n_landcover) — minutes on a domain with a few hundred
  boundary buildings and a city-wide landcover layer. The arithmetic is
  unchanged; only the candidate set is narrowed. The ring intersection is
  also computed once instead of twice.

- `merge`: the `difference` mode only differences the raw features the user
  geometry can actually touch, and `footprint` mode drops them by index
  query. Differencing every raw feature against one large unioned geometry
  was the most expensive operation in the pipeline on a city-wide layer.
  Verified to produce identical output to the previous implementation.

**Features**

- `--log-file PATH` appends the full log to a file, uncoloured and always
  dated. It defaults to `<output_dir>/palm_preproc.log`, so every case
  directory now keeps a record of how it was built — including the
  topology and child-sizing decisions, which are otherwise only visible on
  the terminal that ran it. `--log-file ''` opts out.

- `--version` prints the version and exits.

- `domains_report.txt` carries a provenance header: palm_preproc version,
  config file path, config hash (the same one the resume state uses, so two
  runs differing only in a setting are distinguishable) and the command
  line.

- `tests/`: pytest unit tests for the pure functions — the topology rules
  (`npe`, `valid_pairs`, `best_palm_grid`), grid snapping, the nesting
  validation including the strict-nesting guarantee holding by
  construction over several origins, the namelist substitution, the
  duration parser and the p3d -> p3dr swap. 48 tests, no data needed:

      pytest tests/

  `test.py` remains the end-to-end environment check; these are
  complementary — one proves the environment works, the other proves the
  arithmetic is right.

## 1.1.0

**Correctness**

- `project.state_file` accepts `false` (no state file, no resume) as well as
  `null` (default path) or an explicit path. A boolean previously crashed with
  `TypeError: argument should be a str ... not 'bool'` — which unquoted YAML
  values like `no` or `off` also produce.

- `merge.reassign_id` now **creates** the id column when the source layer does
  not have one, instead of silently doing nothing. A landcover layer without
  `lid` previously produced output without `lid`, which PALM-GeM reports as
  `column "lid" does not exist`. A layer that skips the merge step entirely
  (no user data for it) is now flagged with a warning naming the missing
  column.

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

- A `palmrun` submit script (`submit_<case>.sh`) is generated alongside the
  namelists, with `-X` (total MPI processes across parent and children) and
  `-T` taken from the topology chosen for the case, so the submit script can
  never disagree with the `npex`/`npey` in the `_p3d` files. Wall-clock limit,
  queue, palmrun config and activation string come from `templates.values`;
  disable with `templates.submit: false`.

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
