"""Processor topology enumeration and interactive selection.

The topology rules and the configuration enumeration implemented here are by

    Hynek Reznicek (reznicek@cs.cas.cz)

from his standalone scripts procesor_topology.py and
procesor_topology_nested.py ("Program for finding possible procesor
topology"), integrated into palm_preproc with his rules unchanged:

  per domain (npe / valid_pairs, see steps/domains.py):
    - n divisible by 4 (multigrid psolver), npe in [4, n/4],
      cells/core even and > 9; with opt, cells/core divisible by 4,
    - per-core aspect ratio in (0.5, 2.0), total cores even;
  jointly for nested runs (procesor_topology_nested.py):
    - total cores parent:child between 1:3 and 1:1
      (0.300 < parent/child < 1.001; depends on the domain ratio and
      their vertical layers nz:nz_N02).

When the templates stage needs npex/npey and they are not pinned in the
config, the valid configurations are presented as a numbered list in the
log and the user picks one; in non-interactive contexts (batch jobs, piped
input) the recommended configuration is chosen automatically.
"""

import math
import sys

from ..log import get_logger
from .domains import valid_pairs

log = get_logger()


# ------------------------------
# 1. CONFIGURATION ENUMERATION (rules by Hynek Reznicek)
# ------------------------------
def single_configurations(spec, opt=True):
    """Valid topologies for one domain: list of dicts with npex/npey,
    cells/core and total cores."""
    out = []
    for npex, npey, cx, cy in valid_pairs(spec.width_pts, spec.height_pts, opt):
        out.append({"npex": npex, "npey": npey, "cx": cx, "cy": cy,
                    "cores": npex * npey})
    out.sort(key=lambda c: c["cores"])   # smallest first
    return out


def nested_configurations(parent_spec, child_spec, opt=True, opt_n02=True):
    """Joint parent+child topologies passing Hynek Reznicek's nested rule:
    total cores parent:child between 1:3 and 1:1."""
    parents = single_configurations(parent_spec, opt)
    children = single_configurations(child_spec, opt_n02)
    out = []
    for p in parents:
        for c in children:
            ratio = p["cores"] / c["cores"]
            if 0.300 < ratio < 1.001:
                out.append({"parent": p, "child": c,
                            "sum": p["cores"] + c["cores"]})
    # ordered by total cores (= nodes), smallest first
    out.sort(key=lambda c: (c["sum"], c["parent"]["cores"]))
    return out


# Grid points per core are the measure of work. In a nested run parent and
# child advance together, one time step at a time, so a step takes as long
# as the domain with MORE points per core; the other one waits. Balancing
# core COUNTS (the 1:3 .. 1:1 rule above) does not balance that: palmovka
# was offered parent 432 + child 448 cores with 82 000 vs 156 000 points
# per core, half the parent's cores idle. `pts` is {domain: nx*ny*nz} for
# a nested run, or the single domain's count; None keeps the old ranking.

# A layout costing at most this much more node-hours than the cheapest
# still counts as cheap; among those the fastest is recommended.
COST_SLACK = 0.05


def step_load(c, pts, nested=True):
    """Grid points per core of the slower domain (the time-step cost)."""
    if nested:
        return max(pts["parent"] / c["parent"]["cores"],
                   pts["child"] / c["child"]["cores"])
    return pts / c["cores"]


def _cores(c, nested):
    return c["sum"] if nested else c["cores"]


def _booked(cores, node_cpus):
    """What a job is billed: whole nodes, or plain cores without node_cpus."""
    return math.ceil(cores / node_cpus) if node_cpus else cores


def _squareness(c, nested):
    if nested:
        return (abs(c["parent"]["npex"] - c["parent"]["npey"])
                + abs(c["child"]["npex"] - c["child"]["npey"]))
    return abs(c["npex"] - c["npey"])


def best_per_node_group(configs, node_cpus, nested=True, pts=None):
    """Keep only the best-packed configuration for each whole node count:
    the one that books ceil(cores/node_cpus) nodes with the least idle space
    on the last node (highest fractional node count in that group). E.g. from
    all the 14.xx-node options only 14.95 survives. No-op without node_cpus.

    With `pts`, the fastest layout of each group survives instead: every
    layout in a group books the same nodes, so the one with the lowest
    load per core is simply the best use of them."""
    if not node_cpus:
        return configs
    if pts is not None:
        best = {}
        for c in configs:
            g = _booked(_cores(c, nested), node_cpus)
            key = (step_load(c, pts, nested), _squareness(c, nested))
            if g not in best or key < best[g][0]:
                best[g] = (key, c)
        out = [v[1] for _, v in sorted(best.items())]
        if len(out) < len(configs):
            log.debug(f"[topology] fastest per node count: {len(configs)} "
                      f"-> {len(out)}")
        return out

    def cores(c):
        return c["sum"] if nested else c["cores"]

    import math
    best = {}
    for c in configs:
        nodes = cores(c) / node_cpus
        group = math.ceil(nodes) if nodes % 1 else int(nodes)
        # highest fractional node count in the group = least waste when
        # rounded up (an exact whole number is best of all)
        frac = nodes % 1
        rank = 1.0 if frac == 0 else frac
        if group not in best or rank > best[group][0]:
            best[group] = (rank, c)
    out = [v[1] for _, v in sorted(best.items())]
    if len(out) < len(configs):
        log.debug(f"[topology] best-per-node-group: {len(configs)} -> {len(out)} "
                  f"(one per whole node count)")
    return out


def filter_by_nodes(configs, node_cpus, min_nodes=None, max_nodes=None,
                    nested=True):
    """Keep configurations whose node count (total cores / node_cpus) lies
    within [min_nodes, max_nodes]. No-op without node_cpus or limits."""
    if not node_cpus or not (min_nodes or max_nodes):
        return configs

    def nodes(c):
        return (c["sum"] if nested else c["cores"]) / node_cpus
    kept = [c for c in configs
            if (not min_nodes or nodes(c) >= min_nodes)
            and (not max_nodes or nodes(c) <= max_nodes)]
    log.debug(f"[topology] node limits [{min_nodes or '-'}, {max_nodes or '-'}] @ "
              f"{node_cpus} cpus/node: {len(kept)}/{len(configs)} kept")
    return kept


def dedupe_by_cores(configs, nested=True, pts=None):
    """Keep one configuration per distinct total core count (= per distinct
    node count): the one with the most square per-domain core grids. The
    node count determines the job cost, so listing several layouts that
    cost the same only pads the selection.

    With `pts`, the best-BALANCED split of each core count is kept (lowest
    load per core, then squareness): 880 cores as 432 + 448 and as 288 +
    592 cost the same and are not equally fast."""
    if pts is not None:
        best = {}
        for c in configs:
            k = _cores(c, nested)
            key = (step_load(c, pts, nested), _squareness(c, nested))
            if k not in best or key < best[k][0]:
                best[k] = (key, c)
        out = [v[1] for _, v in sorted(best.items())]
        if len(out) < len(configs):
            log.debug(f"[topology] {len(configs)} topologies collapsed to "
                      f"{len(out)} (best balanced per core count)")
        return out
    def cores(c):
        return c["sum"] if nested else c["cores"]

    def squareness(c):
        if nested:
            return (abs(c["parent"]["npex"] - c["parent"]["npey"])
                    + abs(c["child"]["npex"] - c["child"]["npey"]))
        return abs(c["npex"] - c["npey"])

    best = {}
    for c in configs:
        k = cores(c)
        if k not in best or squareness(c) < squareness(best[k]):
            best[k] = c
    out = sorted(best.values(), key=cores)
    if len(out) < len(configs):
        log.debug(f"[topology] {len(configs)} topologies collapsed to {len(out)} "
                  f"(one per node count)")
    return out


# ------------------------------
# 2. RECOMMENDATION
# ------------------------------
def _closeness(cfg_cores, target):
    return abs(cfg_cores - target) if target else cfg_cores


def _node_waste(cores, node_cpus):
    """Idle fraction of the last booked node. A job runs on
    ceil(cores/node_cpus) whole nodes; this is the unused part of that last
    node: 0.0 at an exact multiple of node_cpus, approaching 1.0 just above a
    multiple. So 14.85 nodes -> 0.15 wasted (book 15), 14.03 -> 0.97 wasted
    (also book 15). Lower = better use of the booked nodes."""
    if not node_cpus:
        return 0.0
    frac = (cores / node_cpus) % 1.0
    return 0.0 if frac == 0.0 else 1.0 - frac


def _recommend_by_load(configs, node_cpus, pts, nested):
    """Index of the fastest layout among those within COST_SLACK of the
    cheapest, cost = booked nodes (or cores) x load per core, i.e.
    node-hours up to a constant."""
    loads = [step_load(c, pts, nested) for c in configs]
    costs = [_booked(_cores(c, nested), node_cpus) * w
             for c, w in zip(configs, loads)]
    cheap = min(costs) * (1 + COST_SLACK)
    return min((i for i, cst in enumerate(costs) if cst <= cheap),
               key=lambda i: (loads[i], _squareness(configs[i], nested)))


def recommend_nested(configs, npes_parent, npes_child, node_cpus=None,
                     pts=None):
    """Index of the recommended nested configuration.

    With core-count targets set, closeness to them leads. Without targets and
    with node_cpus set, the recommendation minimises wasted cores on the last
    booked node (a job is billed ceil(cores/node_cpus) whole nodes, so a node
    count just below an integer packs best); ties break toward more cores,
    then squarer core grids. Without node_cpus, the most cores (fastest run).

    With `pts` (grid points per domain) and no targets: the fastest layout
    among those within COST_SLACK of the cheapest in node-hours, both
    measured by the slower domain's load per core.
    """
    if not configs:
        return None
    target = (npes_parent or 0) + (npes_child or 0)
    if pts is not None and not target:
        return _recommend_by_load(configs, node_cpus, pts, nested=True)

    def key(c):
        sq = (abs(c["parent"]["npex"] - c["parent"]["npey"])
              + abs(c["child"]["npex"] - c["child"]["npey"]))
        waste = round(_node_waste(c["sum"], node_cpus), 6)
        if target:
            return (_closeness(c["sum"], target),
                    _closeness(c["parent"]["cores"], npes_parent or 0), waste, sq)
        if node_cpus:
            return (waste, -c["sum"], sq)
        return (-c["sum"], sq)
    return min(range(len(configs)), key=lambda i: key(configs[i]))


def recommend_single(configs, npes, node_cpus=None, pts=None):
    if not configs:
        return None
    if pts is not None and not npes:
        return _recommend_by_load(configs, node_cpus, pts, nested=False)

    def key(c):
        sq = abs(c["npex"] - c["npey"])
        waste = round(_node_waste(c["cores"], node_cpus), 6)
        if npes:
            return (_closeness(c["cores"], npes), waste, sq)
        if node_cpus:
            return (waste, -c["cores"], sq)
        return (-c["cores"], sq)
    return min(range(len(configs)), key=lambda i: key(configs[i]))


# ------------------------------
# 3. PRESENTATION + INTERACTIVE CHOICE
# ------------------------------
def _fmt_single(c):
    return (f"npex = {c['npex']:>3}, npey = {c['npey']:>3} ({c['cores']:>5}) "
            f" cells/core {c['cx']}x{c['cy']}")


def _log_nested_table(configs, recommended, node_cpus, pts=None):
    log.info("Possible processor topologies "
             "(rules by Hynek Reznicek, procesor_topology_nested.py), "
             "one per node count:")
    fastest = (min(step_load(c, pts) for c in configs) if pts and configs
               else None)
    if pts:
        log.info("(speed: estimated from the grid points per core of the "
                 "slower domain; both domains advance step by step together)")
    for i, c in enumerate(configs, 1):
        nodes = f", nodes = {c['sum'] / node_cpus:.2f}" if node_cpus else ""
        mark = "  <- recommended" if i - 1 == recommended else ""
        log.info(f"[{i:>3}] parent: {_fmt_single(c['parent'])}{mark}")
        log.info(f"      N02:    {_fmt_single(c['child'])}   "
                 f"sum_cpu = {c['sum']}{nodes}")
        if pts:
            wp = pts["parent"] / c["parent"]["cores"]
            wc = pts["child"] / c["child"]["cores"]
            log.info(f"      points/core parent {wp/1e3:,.0f}k, N02 "
                     f"{wc/1e3:,.0f}k -> speed "
                     f"{fastest / max(wp, wc):.2f} of the fastest")


def _log_single_table(configs, recommended, node_cpus, pts=None):
    log.info("Possible processor topologies "
             "(rules by Hynek Reznicek, procesor_topology.py), "
             "one per node count:")
    for i, c in enumerate(configs, 1):
        nodes = f", nodes = {c['cores'] / node_cpus:.2f}" if node_cpus else ""
        load = f", points/core {pts / c['cores'] / 1e3:,.0f}k" if pts else ""
        mark = "  <- recommended" if i - 1 == recommended else ""
        log.info(f"[{i:>3}] {_fmt_single(c)}{nodes}{load}{mark}")


def _prompt_index(n, recommended):
    """Ask the user to pick 1..n; Enter accepts the recommendation."""
    default = f" (Enter = {recommended + 1})" if recommended is not None else ""
    while True:
        try:
            raw = input(f"Select topology [1-{n}]{default}: ").strip()
        except EOFError:
            return recommended
        if not raw and recommended is not None:
            return recommended
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        log.warning(f"Invalid choice '{raw}'; enter a number between 1 and {n}.")


def choose_topology(configs, recommended, mode="ask", node_cpus=None,
                    nested=True, interactive_override=None, pts=None):
    """Present configurations and return the chosen index (or None).

    mode: "ask"  -> interactive prompt when stdin is a terminal,
                    otherwise fall back to the recommendation;
          "auto" -> always take the recommendation silently.
    """
    if not configs:
        log.warning("[topology] no valid processor topology found (relax "
                    "domains.<d>.topology_opt, or widen "
                    "templates.values.min_nodes/max_nodes)")
        return None
    interactive = (interactive_override if interactive_override is not None
                   else sys.stdin.isatty())
    if mode == "auto" or not interactive:
        if recommended is not None and recommended < len(configs):
            why = ("templates.topology_select: false" if mode == "auto"
                   else "non-interactive session")
            c = configs[recommended]
            desc = (f"parent: {_fmt_single(c['parent'])} | "
                    f"N02: {_fmt_single(c['child'])}"
                    if isinstance(c, dict) and "parent" in c
                    else _fmt_single(c))
            # Pin npex/npey in templates.values to choose explicitly; the
            # chosen layout is logged by the caller.
            log.info(f"[topology] auto-selected ({why}) from "
                     f"{len(configs)} valid option(s)")
            log.debug(f"[topology] {desc}")
        return recommended
    (_log_nested_table if nested else _log_single_table)(
        configs, recommended, node_cpus, pts)
    return _prompt_index(len(configs), recommended)
