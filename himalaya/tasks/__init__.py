"""MuJoCo task package.

Import the environment from :mod:`himalaya.tasks.himalaya_env_cfg`. Keeping
this package initializer dependency-light lets config tooling inspect the
curriculum without importing JAX or compiling MuJoCo.
"""

from .g1_cfg import CURRICULUM_SLOPES_DEG

__all__ = ["CURRICULUM_SLOPES_DEG"]
