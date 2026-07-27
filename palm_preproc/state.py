"""JSON resume state for palm_preproc (same idea as palm_postproc).

The state file records which pipeline steps are already done, plus small
pieces of derived data (the computed domain specs) so that later stages can
resume without recomputing. The state is invalidated automatically when the
config hash changes.
"""

import hashlib
import json
from pathlib import Path

from .log import get_logger

log = get_logger()


# ------------------------------
# 1. CONFIG HASH
# ------------------------------
def config_hash(cfg_dict):
    """Stable hash of the (resolved) config dictionary."""
    canon = json.dumps(cfg_dict, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


# ------------------------------
# 2. STATE OBJECT
# ------------------------------
class State:
    def __init__(self, path, cfg_hash, force=False):
        self.path = Path(path)
        self.cfg_hash = cfg_hash
        self._d = {"config_hash": cfg_hash, "done": {}, "data": {}}
        if force:
            log.info("State: --force given, ignoring previous state.")
            return
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text())
            except Exception as exc:  # corrupt state -> start fresh
                log.warning(f"State: could not read {self.path} ({exc}); starting fresh.")
                return
            if loaded.get("config_hash") != cfg_hash:
                log.warning("State: config changed since last run; state invalidated.")
                return
            self._d = loaded
            n = len(self._d.get("done", {}))
            if n:
                log.info(f"State: resuming, {n} step(s) already done.")

    # -- step bookkeeping ------------------------------------------------
    def is_done(self, key):
        return bool(self._d["done"].get(key))

    def mark_done(self, key):
        self._d["done"][key] = True
        self.save()

    # -- derived data (domain specs, priority geometry WKT, ...) ---------
    def get_data(self, key, default=None):
        return self._d["data"].get(key, default)

    def set_data(self, key, value):
        self._d["data"][key] = value
        self.save()

    # -- selective invalidation -----------------------------------------
    def invalidate_prefixes(self, prefixes):
        """Un-mark every done step whose key starts with any of `prefixes`
        (e.g. 'clip:', 'merge:', 'mask:'). Used when the domain geometry
        changes so downstream data is regenerated for the new extent.
        Returns the number of steps cleared."""
        done = self._d["done"]
        victims = [k for k in done
                   if any(k.startswith(p) for p in prefixes)]
        for k in victims:
            del done[k]
        if victims:
            self.save()
        return len(victims)

    # -- persistence ------------------------------------------------------
    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._d, indent=2, default=str))
