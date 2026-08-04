"""Unit tests for the pure functions of palm_preproc.

    pytest tests/

These cover the code where a silent wrong answer is expensive: the
processor-topology rules (a bad topology wastes node-hours), the nesting
validation (a displaced child static driver), the namelist substitution
(a corrupted p3d), and the duration parser. They need no data and no
geospatial stack beyond shapely, so they run in a second and are the right
place to add a regression whenever something breaks.

The end-to-end smoke test lives in ../test.py and is complementary: it
proves the environment works, these prove the arithmetic is right.
"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from palm_preproc.steps.domains import (          # noqa: E402
    DomainSpec, NestingError, best_palm_grid, check_nesting,
    joint_topology_score, make_child_spec, make_parent_spec, npe, snap_up,
    valid_pairs,
)
from palm_preproc.steps.templates import (        # noqa: E402
    _parse_hours, _sub_keyed, make_p3dr,
)


# ------------------------------
# 1. GRID SNAPPING
# ------------------------------
class TestSnapUp:
    @pytest.mark.parametrize("n,stride,want", [
        (1, 8, 8), (8, 8, 8), (9, 8, 16), (100, 8, 104),
        (0, 8, 0), (7.1, 8, 8), (16.0, 8, 16),
    ])
    def test_values(self, n, stride, want):
        assert snap_up(n, stride) == want

    def test_is_a_multiple_and_never_shrinks(self):
        for n in range(1, 400):
            for stride in (2, 4, 8, 16):
                s = snap_up(n, stride)
                assert s % stride == 0
                assert s >= n


# ------------------------------
# 2. TOPOLOGY RULES (Hynek Reznicek)
# ------------------------------
class TestNpe:
    def test_requires_divisibility_by_four(self):
        assert npe(30) == []
        assert npe(31) == []

    def test_pairs_reconstruct_n(self):
        for n in (64, 128, 512, 608):
            for cores, per_core in npe(n):
                assert cores * per_core == n

    def test_documented_constraints_hold(self):
        for n in (64, 128, 256, 512):
            for cores, per_core in npe(n, opt=True):
                assert 4 <= cores <= n // 4        # npe in [4, n/4]
                assert per_core % 2 == 0           # cells/core even
                assert per_core > 9                # and > 9
                assert per_core % 4 == 0           # opt: divisible by 4

    def test_opt_false_is_a_superset(self):
        for n in (64, 128, 512):
            strict = {tuple(p) for p in npe(n, opt=True)}
            loose = {tuple(p) for p in npe(n, opt=False)}
            assert strict <= loose


class TestValidPairs:
    def test_aspect_ratio_and_even_cores(self):
        for npex, npey, cx, cy in valid_pairs(512, 608):
            assert 0.499 < cx / cy < 2.001
            assert (npex * npey) % 2 == 0

    def test_score_matches_pair_count(self):
        assert joint_topology_score(512, 608) == len(valid_pairs(512, 608))

    def test_undecomposable_grid_scores_zero(self):
        assert joint_topology_score(30, 30) == 0


class TestBestPalmGrid:
    def test_never_shrinks_below_the_request(self):
        w, h, _ = best_palm_grid(500, 600, 8, max_overhead=0.10)
        assert w >= 500 and h >= 600

    def test_respects_the_overhead_window(self):
        w, h, _ = best_palm_grid(500, 600, 8, max_overhead=0.10)
        assert w <= 500 * 1.10 + 8 and h <= 600 * 1.10 + 8

    def test_result_is_on_the_stride(self):
        w, h, _ = best_palm_grid(500, 600, 8)
        assert w % 8 == 0 and h % 8 == 0

    def test_is_at_least_as_good_as_the_plain_snap(self):
        w, h, score = best_palm_grid(500, 600, 8)
        base = joint_topology_score(snap_up(500, 8), snap_up(600, 8))
        assert score >= base


# ------------------------------
# 3. DOMAIN SPEC + NESTING
# ------------------------------
def _spec(name, dx, ox, oy, w, h):
    return DomainSpec(name, "EPSG:32633", dx, ox, oy, w, h)


class TestDomainSpec:
    def test_palm_index_convention(self):
        s = _spec("child", 1.0, 0.0, 0.0, 512, 608)
        assert s.nx == 511 and s.ny == 607     # nx = width_pts - 1

    def test_extent_and_corners(self):
        s = _spec("child", 2.0, 100.0, 200.0, 50, 40)
        assert s.width_m == 100.0 and s.height_m == 80.0
        assert s.maxx == 200.0 and s.maxy == 280.0

    def test_roundtrip_through_dict(self):
        s = _spec("parent", 4.0, 1.5, 2.5, 64, 128)
        assert DomainSpec.from_dict(s.to_dict()) == s


class TestCheckNesting:
    def test_aligned_child_passes(self):
        parent = _spec("parent", 4.0, 0.0, 0.0, 200, 200)
        child = _spec("child", 1.0, 40.0, 40.0, 400, 400)
        check_nesting(child, parent, strict=True)      # must not raise

    def test_suborigin_offset_is_rejected(self):
        parent = _spec("parent", 4.0, 0.0, 0.0, 200, 200)
        child = _spec("child", 1.0, 41.5, 40.0, 400, 400)   # 1.5 m off
        with pytest.raises(NestingError):
            check_nesting(child, parent, strict=True)

    def test_extent_off_the_parent_lattice_is_rejected(self):
        parent = _spec("parent", 4.0, 0.0, 0.0, 200, 200)
        child = _spec("child", 1.0, 40.0, 40.0, 401, 400)   # 401 m wide
        with pytest.raises(NestingError):
            check_nesting(child, parent, strict=True)

    def test_child_outside_parent_is_rejected(self):
        parent = _spec("parent", 4.0, 0.0, 0.0, 100, 100)
        child = _spec("child", 1.0, 40.0, 40.0, 400, 400)
        with pytest.raises(NestingError):
            check_nesting(child, parent, strict=True)

    def test_non_strict_downgrades_to_a_warning(self):
        parent = _spec("parent", 4.0, 0.0, 0.0, 200, 200)
        child = _spec("child", 1.0, 41.5, 40.0, 400, 400)
        check_nesting(child, parent, strict=False)     # must not raise


class TestSpecConstruction:
    """The strict-nesting guarantee must hold by construction."""

    @pytest.mark.parametrize("bounds", [
        (0.0, 0.0, 500.0, 600.0),
        (458123.7, 5548211.3, 458611.1, 5548799.9),   # realistic UTM
        (-1000.5, -2000.25, -500.0, -1500.0),
    ])
    def test_child_always_lands_on_the_parent_lattice(self, bounds):
        child_cfg = {"grid_size": 1.0, "buffer": 0.0, "min_power": 3,
                     "optimize_topology": False}
        parent_cfg = {"grid_size": 4.0, "buffer": 400.0, "min_power": 3,
                      "optimize_topology": False}
        child = make_child_spec(bounds, child_cfg, "EPSG:32633",
                                parent_grid_size=4.0, align_to_parent=True)
        parent = make_parent_spec(child, parent_cfg, "EPSG:32633", strict=True)
        check_nesting(child, parent, strict=True)

        off_x = child.origin_x - parent.origin_x
        off_y = child.origin_y - parent.origin_y
        assert math.isclose(off_x % 4.0, 0.0, abs_tol=1e-6)
        assert math.isclose(off_y % 4.0, 0.0, abs_tol=1e-6)
        assert math.isclose(child.width_m % 4.0, 0.0, abs_tol=1e-6)

    def test_parent_contains_child_plus_buffer(self):
        child_cfg = {"grid_size": 1.0, "buffer": 0.0, "min_power": 3,
                     "optimize_topology": False}
        parent_cfg = {"grid_size": 4.0, "buffer": 400.0, "min_power": 3,
                      "optimize_topology": False}
        child = make_child_spec((0.0, 0.0, 500.0, 600.0), child_cfg,
                                "EPSG:32633", parent_grid_size=4.0)
        parent = make_parent_spec(child, parent_cfg, "EPSG:32633")
        assert parent.origin_x <= child.origin_x - 400.0 + 1e-6
        assert parent.maxx >= child.maxx + 400.0 - 1e-6


# ------------------------------
# 4. NAMELIST SUBSTITUTION
# ------------------------------
class TestSubKeyed:
    def test_replaces_the_placeholder(self):
        assert _sub_keyed("nz = <>", "nz", 96) == "nz = 96"

    def test_tolerates_whitespace_variations(self):
        for text in ("nz=<>", "nz  =  <>", "   nz = <>"):
            assert "96" in _sub_keyed(text, "nz", 96)

    def test_does_not_match_the_tail_of_a_longer_key(self):
        # The regression this guards: without a left anchor, key "nz" also
        # fires inside "max_nz" and "dz" inside "ref_dz", silently writing
        # the wrong value into a namelist with no error anywhere.
        assert _sub_keyed("max_nz = <>", "nz", 96) == "max_nz = <>"
        assert _sub_keyed("ref_dz = <>", "dz", 2.0) == "ref_dz = <>"
        assert _sub_keyed("dt_dz = <>", "dz", 2.0) == "dt_dz = <>"

    def test_still_matches_the_real_key_beside_a_longer_one(self):
        text = "max_nz = <>\nnz = <>"
        out = _sub_keyed(text, "nz", 96)
        assert out == "max_nz = <>\nnz = 96"

    def test_suffix_form(self):
        assert _sub_keyed("dx = <>.0", "dx", "2.0", suffix=".0") == "dx = 2.0"

    def test_custom_separator(self):
        assert _sub_keyed("nz: <>", "nz", 96, sep=":") == "nz: 96"


class TestParseHours:
    @pytest.mark.parametrize("value,want", [
        (29, 29.0), (29.5, 29.5), ("29", 29.0), ("29 h", 29.0),
        ("1 d", 24.0), ("90 min", 1.5), ("3600 s", 1.0), ("2 hours", 2.0),
    ])
    def test_accepted_forms(self, value, want):
        assert _parse_hours(value) == pytest.approx(want)

    def test_none_and_garbage(self):
        assert _parse_hours(None) is None
        assert _parse_hours("soon") is None


class TestMakeP3dr:
    def test_swaps_the_initialisation(self):
        p3d = ("&initialization_parameters\n"
               "    initializing_actions = 'inifor',\n"
               "!   initializing_actions = 'read_restart_data',\n"
               "/\n")
        out = make_p3dr(p3d)
        lines = out.splitlines()
        active = [l for l in lines if l.lstrip().startswith("initializing_actions")]
        assert len(active) == 1
        assert "read_restart_data" in active[0]
        assert any(l.startswith("!") and "inifor" in l for l in lines)

    def test_leaves_everything_else_alone(self):
        p3d = "&runtime_parameters\n    end_time = 3600.0,\n/\n"
        assert make_p3dr(p3d) == p3d
