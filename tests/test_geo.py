"""Geometry and raster steps that need the geospatial stack.

    pytest tests/

Skipped when geopandas or rasterio is not installed; test.py covers the
same stack end to end.
"""

import sys
from pathlib import Path

import pytest

gpd = pytest.importorskip("geopandas")
rasterio = pytest.importorskip("rasterio")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                        # noqa: E402
from rasterio.transform import from_origin                # noqa: E402
from shapely.geometry import LineString, Point, box       # noqa: E402

from palm_preproc.steps.boundary import _walls_under      # noqa: E402
from palm_preproc.steps.clip import merge_raster_task     # noqa: E402
from palm_preproc.steps.domains import DomainSpec         # noqa: E402


class TestWallMatching:
    REMOVED = box(0, 0, 10, 10)
    KEPT = box(10, 0, 20, 10)          # shares the edge x = 10

    def _match(self, *walls):
        return list(_walls_under(gpd.GeoSeries(list(walls)),
                                 self.REMOVED, self.KEPT))

    def test_line_walls_on_a_removed_outline_go(self):
        # a line on the outline lies on the footprint BOUNDARY, where
        # `within` is false: these walls all survived before
        assert self._match(LineString([(0, 0), (10, 0)]),
                           LineString([(0, 0), (0, 10)])) == [True, True]

    def test_shared_and_kept_walls_stay(self):
        assert self._match(LineString([(10, 0), (10, 10)]),
                           LineString([(10, 0), (20, 0)])) == [False, False]

    def test_polygon_and_point_walls(self):
        ring = self.REMOVED.difference(box(1, 1, 9, 9))
        assert self._match(ring, Point(5, 0), Point(15, 0)) == [
            True, True, False]


def _raster(path, arr, x0, y0, res):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1],
                       height=arr.shape[0], count=1, dtype="float32",
                       crs="EPSG:32633", nodata=-9999.0,
                       transform=from_origin(x0, y0 + arr.shape[0] * res,
                                             res, res)) as ds:
        ds.write(arr.astype("float32"), 1)


class TestBuildingsMask:
    def test_the_layer_mask_bounds_the_user_raster(self, tmp_path):
        # merge.masks.buildings was accepted and then ignored: the raster
        # merge always used the shared domain footprint
        spec = DomainSpec("child", "EPSG:32633", 1.0, 0.0, 0.0, 16, 16)
        _raster(tmp_path / "raw.tif", np.full((16, 16), 5.0), 0, 0, 1.0)
        _raster(tmp_path / "user.tif", np.full((16, 16), 30.0), 0, 0, 1.0)
        everywhere = box(0, 0, 16, 16).wkt
        west_half = box(0, 0, 8, 16).wkt
        merge_raster_task(str(tmp_path / "user.tif"), str(tmp_path / "raw.tif"),
                          str(tmp_path / "out.tif"), spec.to_dict(),
                          everywhere, "nearest", -9999.0, west_half)
        with rasterio.open(tmp_path / "out.tif") as ds:
            out = ds.read(1)
        assert (out[:, :8] == 30.0).all()     # user inside the mask
        assert (out[:, 9:] == 5.0).all()      # raw outside it
