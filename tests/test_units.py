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
    _PLACEHOLDER_RE, _parse_hours, _pgem_points_stale, _p3d_values,
    _resolve_topology, _stale_reason, _sub_keyed, child_nz, fill_p3d,
    fill_pgem, fmt_real, make_p3dr,
)
from palm_preproc.layout import to_new_names     # noqa: E402


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

    @pytest.mark.parametrize("optimize", [False, True])
    @pytest.mark.parametrize("dx,parent_dx,buffer", [
        (1.0, 4.0, 0.0), (1.0, 3.0, 0.0), (2.0, 4.0, 10.0), (0.5, 2.0, 0.0),
    ])
    def test_child_covers_the_input_and_is_centred(self, dx, parent_dx,
                                                   buffer, optimize):
        # The regression this guards: the origin was centred and THEN
        # floored onto the parent lattice, which shifted the rectangle up
        # to one parent cell west/south and left a strip of the area of
        # interest outside the child in ~3 of 4 random cases.
        import random
        rng = random.Random(7)
        cfg = {"grid_size": dx, "buffer": buffer, "min_power": 3,
               "optimize_topology": optimize, "confirm_optimized": False}
        span = 300 if optimize else 1500     # the size search is costly
        for _ in range(100):
            x0, y0 = rng.uniform(4e5, 5e5), rng.uniform(5.5e6, 5.6e6)
            b = (x0, y0, x0 + rng.uniform(100, span), y0 + rng.uniform(100, span))
            c = make_child_spec(b, cfg, "EPSG:32633",
                                parent_grid_size=parent_dx)
            west, east = b[0] - buffer - c.origin_x, c.maxx - (b[2] + buffer)
            south, north = b[1] - buffer - c.origin_y, c.maxy - (b[3] + buffer)
            assert min(west, east, south, north) >= -1e-6
            assert abs(west - east) <= parent_dx + 1e-6
            assert abs(south - north) <= parent_dx + 1e-6
            assert math.isclose(c.origin_x / parent_dx,
                                round(c.origin_x / parent_dx), abs_tol=1e-6)

    def test_the_documented_case(self):
        cfg = {"grid_size": 1.0, "buffer": 0.0, "min_power": 3,
               "optimize_topology": False}
        c = make_child_spec((1.0, 1.0, 801.0, 801.0), cfg, "EPSG:32633",
                            parent_grid_size=4.0)
        assert c.origin_x <= 1.0 and c.maxx >= 801.0


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


class TestGridDimensions:
    """The static driver PALM-GeM builds from pgem_*.yaml must have exactly
    the nx+1 x ny+1 points the _p3d declares. The regression this guards:
    pgem got the _p3d's nx (last index), PALM-GeM read it as a point
    count, the driver came out one column short, and PALM aborted with
    PAC0333 (static driver x = 766, model 767)."""

    def _ctx(self, spec):
        return {"case": "palmovka", "spec": spec, "other": None, "nz": 504,
                "npex": None, "npey": None, "origin_time": None,
                "hours": None, "seasonal": {}}

    def test_pgem_gets_point_counts_p3d_gets_last_index(self):
        spec = _spec("parent", 4.0, 459000.0, 5546000.0, 768, 768)
        pgem = fill_pgem("nx: <>\nny: <>\n", self._ctx(spec))
        p3d = fill_p3d("nx = <>, ny = <>,\n", self._ctx(spec))
        assert pgem == "nx: 768\nny: 768\n"
        assert p3d == "nx = 767, ny = 767,\n"

    def test_pgem_centre_and_points_reproduce_the_rectangle(self):
        spec = _spec("child", 1.0, 459800.0, 5546800.0, 2016, 1536)
        pgem = fill_pgem("nx: <>\nny: <>\ncent_x: <>\ncent_y: <>\n",
                         self._ctx(spec))
        vals = dict(line.split(": ") for line in pgem.splitlines())
        nx, ny = int(vals["nx"]), int(vals["ny"])
        assert float(vals["cent_x"]) - nx * spec.grid_size / 2 == spec.origin_x
        assert float(vals["cent_y"]) - ny * spec.grid_size / 2 == spec.origin_y


class TestNumberFormatting:
    @pytest.mark.parametrize("value,want", [
        (1, "1.0"), (2.0, "2.0"), (0.25, "0.25"), (1.25, "1.25"),
        (400.0, "400.0"), (1002.5, "1002.5"),
    ])
    def test_reals_keep_their_digits(self, value, want):
        # ':.1f' wrote dx = 0.25 as 0.2 into the namelist
        assert fmt_real(value) == want

    def test_p3d_grid_spacing(self):
        spec = _spec("child", 0.25, 0.0, 0.0, 64, 64)
        ctx = {"case": "c", "spec": spec, "other": None, "nz": None,
               "npex": None, "npey": None, "origin_time": None,
               "hours": None, "seasonal": {}}
        assert fill_p3d("dx = <>.0, dz = <>.0,", ctx) == "dx = 0.25, dz = 0.25,"

    def test_placeholders_are_counted_once(self):
        assert len(_PLACEHOLDER_RE.findall("nx = <>, ny = <>, <case>")) == 3


class TestMessageNames:
    def test_prefix_keys_need_a_setting_after_them(self):
        assert to_new_names("nothing to merge.") == "nothing to merge."
        assert to_new_names("nothing to clip.") == "nothing to clip."

    def test_settings_are_still_renamed(self):
        assert to_new_names("merge.mode") == "advanced.merge.mode"
        assert (to_new_names("templates.values.npex_child = 7")
                == "advanced.cpu_topology.npex_child = 7")


# ------------------------------
# 5. NZ AND TOPOLOGY CHOICES
# ------------------------------
class TestChildNz:
    def test_the_mean_governs_ordinary_blocks(self):
        nz, _, rule = child_nz(20.0, 25.0, 1.0, 4.0, 1.5, 8)
        assert (nz, rule) == (80, "mean")

    def test_a_tall_tower_raises_the_top(self):
        # 100 m tower among 15 m blocks: the mean alone gave 64 m
        nz, raw, rule = child_nz(16.0, 100.0, 1.0, 4.0, 1.5, 8)
        assert rule == "tallest"
        assert nz * 1.0 >= 150.0 and nz % 8 == 0


def _topo_cfg(**values):
    tv = {"node_cpus": 80, "min_nodes": 2, "max_nodes": 16}
    tv.update(values)
    return {"templates": {"values": tv, "topology_select": False},
            "domains": {"parent": {"topology_opt": True},
                        "child": {"topology_opt": True}}}


class TestTopologyPins:
    P = DomainSpec("parent", "EPSG:32633", 4.0, 0.0, 0.0, 768, 768)
    C = DomainSpec("child", "EPSG:32633", 1.0, 400.0, 400.0, 2016, 2016)

    def _resolve(self, **pins):
        cfg = _topo_cfg(**pins)
        _resolve_topology(self.C, self.P, cfg, nested=True)
        return cfg["templates"]["values"]

    def test_a_partial_pin_is_kept(self):
        # it used to be overwritten by the chooser without a word
        tv = self._resolve(npex_parent=8)
        assert tv["npex_parent"] == 8
        assert all(tv[k] for k in ("npey_parent", "npex_child", "npey_child"))

    def test_a_pin_that_does_not_divide_the_grid_is_an_error(self):
        with pytest.raises(RuntimeError, match="does not divide"):
            self._resolve(npex_child=5)

    def test_pins_no_topology_agrees_with_are_an_error(self):
        with pytest.raises(RuntimeError, match="No valid topology"):
            self._resolve(npex_parent=24, npey_parent=4)

    def test_full_pins_bypass_the_chooser(self):
        tv = self._resolve(npex_parent=8, npey_parent=8, npex_child=12,
                           npey_child=12)
        assert (tv["npex_parent"], tv["npex_child"]) == (8, 12)


# ------------------------------
# 6. STALE GENERATED FILES
# ------------------------------
class TestStaleSet:
    P = DomainSpec("parent", "EPSG:32633", 4.0, 1000.0, 2000.0, 144, 144)
    C = DomainSpec("child", "EPSG:32633", 2.0, 1200.0, 2200.0, 80, 80)

    P3D = ("!-- nx=63\n &nesting_parameters\n"
           "    domain_layouts = 'parent',   1,  -1, 16,  0.0, 0.0,\n"
           "                     'child',    2,   1, 16, 40.0, 40.0,\n /\n"
           "         nx = 63, ny = 63, nz = 504,\n"
           "         npex = 4,\n         npey = 4,\n")

    def test_values_are_read_back_from_a_p3d(self, tmp_path):
        (tmp_path / "c_p3d").write_text(self.P3D)
        v = _p3d_values(tmp_path / "c_p3d")
        assert (v["nx"], v["ny"], v["nz"], v["npex"], v["npey"]) == (
            63, 63, 504, 4, 4)
        assert v["llx_lly"] == (40.0, 40.0)

    def test_a_changed_parent_makes_a_legacy_set_stale(self):
        # the user's case: buffer 40 -> 200 m, files from the 40 m run kept
        rec = {"legacy": True, "topology": {}, "nz": {},
               "domains": {"parent": {"width_pts": 64, "height_pts": 64}},
               "llx_lly": (40.0, 40.0)}
        assert "parent grid 64x64 -> 144x144" in _stale_reason(
            rec, {"parent": self.P, "child": self.C}, True, "parent")

    def test_a_matching_record_is_not_stale(self):
        from palm_preproc.steps.templates import _geometry
        rec = {"nested": True, "domains": {"parent": _geometry(self.P),
                                           "child": _geometry(self.C)}}
        assert _stale_reason(rec, {"parent": self.P, "child": self.C},
                             True, "parent") is None
        moved = DomainSpec("child", "EPSG:32633", 2.0, 1204.0, 2200.0, 80, 80)
        assert _stale_reason(rec, {"parent": self.P, "child": moved},
                             True, "parent") == "child domain geometry changed"
        del rec["domains"]["child"]["origin_x"]       # incomplete record
        assert _stale_reason(rec, {"parent": self.P, "child": self.C},
                             True, "parent") == "child domain geometry changed"

    def test_a_pre_1_3_1_pgem_is_stale(self, tmp_path):
        f = tmp_path / "pgem_c.yaml"
        f.write_text("domain:\n    nx: 143\n    ny: 143\n")
        assert "points - 1" in _pgem_points_stale(f, self.P)
        f.write_text("domain:\n    nx: 144   # grid POINTS\n    ny: 144\n")
        assert _pgem_points_stale(f, self.P) is None


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
