"""PALM domain rectangle generation (refactor of gen_bounding_rectangle_4.py).

Importable functions used by the palm_preproc pipeline, plus a standalone CLI
(`python -m palm_preproc.domains ...`) that replicates the original script's
behaviour for one-off use.

Improvements over the standalone script:
  * The PARENT domain is derived from the (already snapped) CHILD rectangle
    plus a buffer -- not from the raw input shapefile again. This guarantees
    the parent fully contains the child + buffer.
  * Child/parent nesting alignment: when the parent/child grid-size ratio is
    an integer, the child origin is snapped onto the PARENT grid and the
    child extent (in metres) is forced to a multiple of the parent grid size,
    so the child edges lie exactly on parent grid lines (PALM nesting rule).
  * Topology-aware sizing (procesor_topology_nested.py rules) kept as-is and
    available per-domain.

PALM indexing note: namelist nx/ny are the LAST index, i.e.
    nx = width_pts - 1,  ny = height_pts - 1.
Topology rules operate on cell counts (= width_pts/height_pts here), matching
procesor_topology_nested.py.
"""

import math
from dataclasses import dataclass, asdict

from shapely.geometry import box

from ..log import get_logger

log = get_logger()


class NestingError(RuntimeError):
    """Child/parent nesting geometry violates PALM constraints."""


# ------------------------------
# 1. GRID SIZE HELPERS
# ------------------------------
def snap_up(n, stride):
    """Smallest integer >= n divisible by stride."""
    n = int(math.ceil(n))
    return int(math.ceil(n / stride) * stride)


def _lcm(a, b):
    return abs(a * b) // math.gcd(a, b)


# ------------------------------
# 2. TOPOLOGY RULES (faithful to procesor_topology_nested.py)
# ------------------------------
def npe(n, opt=True):
    """Valid [npe, cells_per_core] pairs for one dimension of n cells.

    Mirrors npe() in procesor_topology_nested.py: n divisible by 4; npe in
    [4, n/4]; cells/core even and > 9. With opt=True, additionally require
    cells/core divisible by 4 (best for multigrid).
    """
    if n % 4 != 0:
        return []
    out = []
    for i in reversed(range(4, n // 4 + 1)):
        if n % i == 0:
            div = n // i
            if div % 2 == 0 and div > 9:
                if opt and div % 4 != 0:
                    continue
                out.append([i, div])
    return out


def valid_pairs(nx, ny, opt=True):
    """Joint (npex, npey) decompositions passing the parent-domain
    constraints: per-core aspect ratio in (0.5, 2.0) and even total cores.
    Returns list of (npex, npey, cells_x, cells_y)."""
    pairs = []
    for (ix, cx) in npe(nx, opt):
        for (iy, cy) in npe(ny, opt):
            ratio = cx / cy
            if not (0.499 < ratio < 2.001):
                continue
            if (ix * iy) % 2 != 0:
                continue
            pairs.append((ix, iy, cx, cy))
    return pairs


def joint_topology_score(nx, ny, opt=True):
    """Number of valid joint decompositions for (nx, ny). More = more flexible."""
    return len(valid_pairs(nx, ny, opt))


def best_palm_grid(width_pts, height_pts, stride, max_overhead=0.10, opt=True):
    """Pick (width_pts, height_pts) maximising joint decomposition flexibility.

    Both axes are snapped up to multiples of `stride` and searched within
    +max_overhead of their minimum size. Ties break toward smaller total size,
    then toward the more square domain. Falls back to the simple snap when no
    decomposable pair exists in the window.
    Returns (width_pts, height_pts, joint_score).
    """
    def axis_candidates(n_pts):
        base = snap_up(n_pts, stride)
        limit = n_pts * (1 + max_overhead)
        cands, c = [], base
        while c <= limit:
            cands.append(c)
            c += stride
        return cands or [base]

    best = None  # (key, w, h, score)
    for w in axis_candidates(width_pts):
        for h in axis_candidates(height_pts):
            score = joint_topology_score(w, h, opt)
            key = (score, -(w + h), -abs(w - h))
            if best is None or key > best[0]:
                best = (key, w, h, score)

    if best is None or best[3] == 0:
        w = snap_up(width_pts, stride)
        h = snap_up(height_pts, stride)
        return w, h, joint_topology_score(w, h, opt)
    return best[1], best[2], best[3]


# ------------------------------
# 3. DOMAIN SPEC
# ------------------------------
@dataclass
class DomainSpec:
    """Fully snapped PALM domain rectangle in the aligned CRS."""
    name: str            # "child" / "parent"
    crs: str             # aligned CRS, e.g. "EPSG:32633"
    grid_size: float     # metres per grid point
    origin_x: float      # SW corner (snapped)
    origin_y: float
    width_pts: int
    height_pts: int
    joint_score: int = -1   # topology score (-1 = not evaluated)

    # -- derived -----------------------------------------------------------
    @property
    def width_m(self):
        return self.width_pts * self.grid_size

    @property
    def height_m(self):
        return self.height_pts * self.grid_size

    @property
    def maxx(self):
        return self.origin_x + self.width_m

    @property
    def maxy(self):
        return self.origin_y + self.height_m

    @property
    def nx(self):
        return self.width_pts - 1

    @property
    def ny(self):
        return self.height_pts - 1

    def rectangle(self):
        return box(self.origin_x, self.origin_y, self.maxx, self.maxy)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_gdf(self, output_crs=None):
        """GeoDataFrame with the same attribute table as gen_bounding_rectangle."""
        import geopandas as gpd
        gdf = gpd.GeoDataFrame(
            {
                "id":         [1],
                "name":       [self.name],
                "grid_size":  [self.grid_size],
                "width_m":    [self.width_m],
                "height_m":   [self.height_m],
                "width_pts":  [self.width_pts],
                "height_pts": [self.height_pts],
                "nx":         [self.nx],
                "ny":         [self.ny],
                "origin_x":   [self.origin_x],   # aligned-CRS SW corner
                "origin_y":   [self.origin_y],
            },
            geometry=[self.rectangle()],
            crs=self.crs,
        )
        if output_crs is not None and str(output_crs) != str(self.crs):
            log.info(f"[{self.name}] reprojecting rectangle to {output_crs} "
                     f"(rotated quadrilateral; do NOT take .total_bounds downstream)")
            gdf = gdf.to_crs(output_crs)
        return gdf


# ------------------------------
# 4. SIZING / SNAPPING CORE
# ------------------------------
def _size_axes(width_pts, height_pts, stride, dcfg):
    """Snap both axes, optionally topology-aware. Returns (w, h, score)."""
    if dcfg.get("optimize_topology"):
        w, h, score = best_palm_grid(
            width_pts, height_pts, stride,
            dcfg.get("topology_max_overhead", 0.10),
            dcfg.get("topology_opt", True),
        )
        base_w = snap_up(width_pts, stride)
        base_h = snap_up(height_pts, stride)
        base_score = joint_topology_score(base_w, base_h, dcfg.get("topology_opt", True))
        log.debug(f"topology-aware sizing: baseline {base_w}x{base_h} "
                 f"(score {base_score}) -> chosen {w}x{h} (score {score})")
        pairs = valid_pairs(w, h, dcfg.get("topology_opt", True))
        if pairs:
            ex = max(pairs, key=lambda p: p[0] * p[1])
            log.debug(f"sample decomposition: npex={ex[0]} npey={ex[1]} "
                     f"(cores={ex[0]*ex[1]}, cells/core={ex[2]}x{ex[3]})")
        return w, h, score
    w = snap_up(width_pts, stride)
    h = snap_up(height_pts, stride)
    return w, h, -1


def make_child_spec(bounds, child_cfg, aligned_crs, parent_grid_size=None,
                    align_to_parent=True, interactive_override=None):
    """Child domain rectangle from input bounds (minx, miny, maxx, maxy).

    The rectangle is expanded symmetrically around the input centre, snapped
    up to a PALM-friendly size, and its origin floored to the alignment grid
    (the parent grid when alignment is on, else the child grid).
    """
    dx = float(child_cfg["grid_size"])
    buffer = float(child_cfg.get("buffer", 0.0))
    stride = 1 << int(child_cfg.get("min_power", 3))

    # Nesting alignment: child extent must be a multiple of the parent grid
    # size, i.e. width_pts divisible by the (integer) grid ratio.
    align_grid = dx
    if align_to_parent and parent_grid_size:
        ratio = parent_grid_size / dx
        if abs(ratio - round(ratio)) < 1e-9:
            stride = _lcm(stride, int(round(ratio)))
            align_grid = parent_grid_size
        else:
            log.warning(f"[domains] child: non-integer parent/child grid ratio {ratio:g}; "
                        f"skipping nesting alignment.")

    minx, miny, maxx, maxy = bounds
    width = (maxx - minx) + 2 * buffer
    height = (maxy - miny) + 2 * buffer
    log.debug(f"child: input bounds {maxx-minx:.1f} x {maxy-miny:.1f} m, "
             f"buffer {buffer:g} m/side, grid {dx:g} m, stride {stride} pts")

    w_pts, h_pts, score = _size_axes(width / dx, height / dx, stride, child_cfg)

    # Remark: the user decides whether an optimization-grown child is
    # accepted (interactive sessions only; batch runs proceed and say so).
    if child_cfg.get("optimize_topology") and child_cfg.get("confirm_optimized", True):
        base_w = snap_up(width / dx, stride)
        base_h = snap_up(height / dx, stride)
        if (w_pts, h_pts) != (base_w, base_h):
            w_pts, h_pts, score = _confirm_optimized_child(
                (w_pts, h_pts, score), (base_w, base_h),
                child_cfg, interactive_override)

    # Centre-expand, then floor SW origin onto the alignment grid. The
    # origin is constructed as (integer lattice index) * align_grid and
    # rounded, so it lies EXACTLY on the lattice by construction.
    cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
    ox = round(math.floor((cx - w_pts * dx / 2) / align_grid) * align_grid, 6)
    oy = round(math.floor((cy - h_pts * dx / 2) / align_grid) * align_grid, 6)

    spec = DomainSpec("child", str(aligned_crs), dx, ox, oy, w_pts, h_pts, score)
    _log_spec(spec)
    return spec


def _confirm_optimized_child(chosen, baseline, child_cfg, interactive_override):
    import sys
    w, h, score = chosen
    bw, bh = baseline
    bscore = joint_topology_score(bw, bh, child_cfg.get("topology_opt", True))
    log.info(f"child: optimization grew the domain "
             f"{bw}x{bh} -> {w}x{h} pts "
             f"(+{w - bw} x +{h - bh}; joint score {bscore} -> {score})")
    interactive = (interactive_override if interactive_override is not None
                   else sys.stdin.isatty())
    if not interactive:
        log.info(f"child: non-interactive session - auto-accepted the "
                 f"optimized size {w}x{h} (baseline was {bw}x{bh}; set "
                 f"domains.child.optimize_topology: false to keep the "
                 f"baseline, or confirm_optimized: false to silence).")
        return chosen
    while True:
        try:
            raw = input(f"Proceed with the optimized child {w}x{h} instead "
                        f"of the baseline {bw}x{bh}? [Y/n]: ").strip().lower()
        except EOFError:
            return chosen
        if raw in ("", "y", "yes"):
            return chosen
        if raw in ("n", "no"):
            log.info(f"child: using the baseline size {bw}x{bh}")
            return bw, bh, bscore
        log.warning(f"Invalid answer '{raw}'; y or n.")


def make_parent_spec(child, parent_cfg, aligned_crs, strict=True):
    """Parent domain: child rectangle + buffer, snapped to the parent grid.

    Origin is floored (SW) to the parent grid below (child SW - buffer); the
    size is snapped up from there, so containment of child+buffer is
    guaranteed and the child offset from the parent origin is an exact
    multiple of the parent grid size.
    """
    dx = float(parent_cfg["grid_size"])
    buffer = float(parent_cfg.get("buffer", 0.0))
    stride = 1 << int(parent_cfg.get("min_power", 3))

    ox = round(math.floor((child.origin_x - buffer) / dx) * dx, 6)
    oy = round(math.floor((child.origin_y - buffer) / dx) * dx, 6)
    req_w_pts = (child.maxx + buffer - ox) / dx
    req_h_pts = (child.maxy + buffer - oy) / dx
    log.debug(f"parent: buffer {buffer:g} m around child, grid {dx:g} m, "
             f"stride {stride} pts")

    w_pts, h_pts, score = _size_axes(req_w_pts, req_h_pts, stride, parent_cfg)

    spec = DomainSpec("parent", str(aligned_crs), dx, ox, oy, w_pts, h_pts, score)
    _log_spec(spec)
    check_nesting(child, spec, strict=strict)
    return spec


def _log_spec(s):
    log.info(f"{s.name}: {s.width_pts} x {s.height_pts} pts @ {s.grid_size:g} m, "
             f"PALM nx={s.nx} ny={s.ny}, origin ({s.origin_x:.2f}, {s.origin_y:.2f})")


def check_nesting(child, parent, strict=True, tol=1e-6):
    """Validate PALM nesting geometry: containment, and child origin offset
    and extent EXACTLY on the parent lattice (within `tol` metres).

    With strict=True (default) any violation raises NestingError so the
    pipeline fails loudly instead of producing a static driver whose content
    is displaced sub-grid from where PALM will place the nest. Set
    domains.strict_nesting: false in the config to downgrade to warnings
    (only sensible for deliberately non-nested setups).
    """
    problems = []
    if not (parent.origin_x <= child.origin_x and parent.origin_y <= child.origin_y
            and child.maxx <= parent.maxx and child.maxy <= parent.maxy):
        problems.append("child rectangle not contained in parent")
    for label, off in (("x", child.origin_x - parent.origin_x),
                       ("y", child.origin_y - parent.origin_y)):
        residual = abs(off / parent.grid_size - round(off / parent.grid_size))
        if residual * parent.grid_size > tol:
            problems.append(
                f"child origin offset in {label} ({off:.6f} m) is off the "
                f"parent lattice ({parent.grid_size:g} m) by "
                f"{residual * parent.grid_size:.6f} m")
    for label, ext in (("width", child.width_m), ("height", child.height_m)):
        residual = abs(ext / parent.grid_size - round(ext / parent.grid_size))
        if residual * parent.grid_size > tol:
            problems.append(
                f"child {label} ({ext:.6f} m) is not a multiple of the "
                f"parent grid ({parent.grid_size:g} m)")
    if not problems:
        log.debug("nesting: child aligns with the parent grid, OK")
        return
    for p in problems:
        log.error(f"nesting: {p}")
    if strict:
        raise NestingError(
            "Nesting geometry invalid: " + "; ".join(problems) +
            ". The child static driver would be displaced relative to the "
            "nest position PALM derives from domain_layouts. Fix the domain "
            "configuration (or set domains.strict_nesting: false to only "
            "warn)."
        )
    log.warning("nesting: continuing despite the problems above "
                "(strict_nesting: false).")


# ------------------------------
# 5. STANDALONE CLI (replacement for gen_bounding_rectangle_4.py)
# ------------------------------
def _cli():
    import argparse
    import sys
    import geopandas as gpd

    p = argparse.ArgumentParser(
        description="Create a UTM-aligned, PALM-snapped bounding rectangle "
                    "from a shapefile (standalone gen_bounding_rectangle).")
    p.add_argument("input", help="input shapefile")
    p.add_argument("output", help="output shapefile")
    p.add_argument("--aligned-crs", default="EPSG:32633")
    p.add_argument("--output-crs", default=None,
                   help="CRS of the written rectangle (default: aligned CRS)")
    p.add_argument("--grid-size", type=float, default=1.0)
    p.add_argument("--buffer", type=float, default=0.0)
    p.add_argument("--min-power", type=int, default=3)
    p.add_argument("--optimize-topology", action="store_true")
    p.add_argument("--max-overhead", type=float, default=0.10)
    p.add_argument("--no-opt", action="store_true",
                   help="drop the cells/core %% 4 == 0 requirement")
    a = p.parse_args()

    from ..log import setup_logging
    setup_logging()

    gdf = gpd.read_file(a.input)
    if gdf.crs is None:
        log.error("Input has no CRS."); sys.exit(1)
    bounds = gdf.to_crs(a.aligned_crs).total_bounds

    dcfg = {
        "grid_size": a.grid_size, "buffer": a.buffer, "min_power": a.min_power,
        "optimize_topology": a.optimize_topology,
        "topology_max_overhead": a.max_overhead, "topology_opt": not a.no_opt,
    }
    spec = make_child_spec(bounds, dcfg, a.aligned_crs, align_to_parent=False)
    out_gdf = spec.to_gdf(a.output_crs or a.aligned_crs)
    from pathlib import Path
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    out_gdf.to_file(a.output)
    log.info(f"Written: {a.output}")


if __name__ == "__main__":
    _cli()
