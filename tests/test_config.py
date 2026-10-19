"""Config loading: the layered defaults and the new-layout translation.

    pytest tests/

Needs only PyYAML: the config reads no geodata, so the files it checks for
existence are empty placeholders.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from palm_preproc.config import Config, ConfigError   # noqa: E402


@pytest.fixture
def root(tmp_path):
    """A project tree with an empty landcover layer and no domain.shp."""
    (tmp_path / "proj" / "DATA_user").mkdir(parents=True)
    (tmp_path / "proj" / "DATA_user" / "landcover.shp").touch()
    return tmp_path


def _load(root, text, name="p.yaml"):
    path = root / name
    path.write_text(f"project:\n    name: proj\n    root: {root}\n" + text)
    return Config(path)


USER = """input:
    user_data:
        dir: ./proj/DATA_user
        domain: null
        domain_from: landcover
"""


class TestExplicitNull:
    def test_domain_null_selects_domain_from(self, root):
        # null used to read as "not set", so the built-in domain.shp
        # default stayed and the run stopped with "Domain shapefile not
        # found" (config/example_minimal.yaml, config/palmovka.yaml)
        cfg = _load(root, USER)
        assert cfg.domain_source()[1] == "extent"

    def test_null_overrides_a_site_default(self, root):
        cfg = _load(root, USER + "cluster:\n    node_cpus: null\n"
                    "    nodes: [null, 8]\n")
        tv = cfg["templates"]["values"]
        assert tv["node_cpus"] is None          # defaults/default.yaml: 80
        assert (tv["min_nodes"], tv["max_nodes"]) == (None, 8)

    def test_null_on_a_required_setting_keeps_the_default(self, root):
        cfg = _load(root, USER + "advanced:\n    grid:\n"
                    "        strict_nesting: null\n")
        assert cfg["domains"]["strict_nesting"] is True

    def test_a_missing_domain_names_the_fix(self, root):
        text = USER.replace("        domain: null\n", "")
        with pytest.raises(ConfigError, match="domain: null"):
            _load(root, text)


class TestPaths:
    def test_a_tilde_root_is_the_home_folder(self, root, monkeypatch):
        # `~/palm/DATA` used to resolve to <config dir>/~/palm/DATA, so no
        # user layer was found and domain_from failed with "found: []"
        monkeypatch.setenv("HOME", str(root.parent))
        cfg_dir = root / "config"
        cfg_dir.mkdir()
        path = cfg_dir / "p.yaml"
        path.write_text(f"project:\n    name: proj\n    root: ~/{root.name}/\n"
                        + USER)
        cfg = Config(path)
        assert cfg["project"]["root"] == str(root)
        assert cfg.domain_source()[1] == "extent"

    def test_a_missing_root_is_named(self, root):
        path = root / "p.yaml"
        path.write_text("project:\n    name: proj\n    root: nowhere\n" + USER)
        with pytest.raises(ConfigError, match="project.root not found"):
            Config(path)

    def test_a_missing_domain_from_layer_names_the_path(self, root):
        (root / "proj" / "DATA_user" / "landcover.shp").unlink()
        with pytest.raises(ConfigError, match=r"DATA_user/landcover\.shp does not exist"):
            _load(root, USER)


class TestSiteDefaults:
    def test_defaults_false_skips_the_site_file(self, root):
        # it used to be read before translation, i.e. never
        cfg = _load(root, "    defaults: false\n" + USER)
        assert cfg["domains"]["parent"]["grid_size"] == 2.0    # built-in
        assert cfg["templates"]["values"]["node_cpus"] is None

    def test_defaults_path_is_used(self, root):
        (root / "site.yaml").write_text("domains:\n  parent:\n"
                                        "    grid_size: 8.0\n")
        cfg = _load(root, "    defaults: site.yaml\n" + USER)
        assert cfg["domains"]["parent"]["grid_size"] == 8.0

    def test_a_missing_defaults_file_is_an_error(self, root):
        with pytest.raises(ConfigError, match="defaults file not found"):
            _load(root, "    defaults: nope.yaml\n" + USER)

    def test_the_standard_file_applies_otherwise(self, root):
        cfg = _load(root, USER)
        assert cfg["domains"]["parent"]["grid_size"] == 4.0    # default.yaml


class TestMasks:
    def test_a_dem_mask_is_rejected(self, root):
        (root / "proj" / "DATA_user" / "mask.shp").touch()
        with pytest.raises(ConfigError, match="user-supplied layers only"):
            _load(root, USER + "advanced:\n    merge:\n"
                  "        masks: {dem: mask.shp}\n")
