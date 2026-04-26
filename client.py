"""Top-level `client` shim for OpenEnv canonical layout.

Real implementation lives at `viveka.client`. This module re-exports
`VivekaClient` so that:

  * `openenv push --validate` finds `client.py` at repo root
  * `pip install git+https://huggingface.co/spaces/...` followed by
    `from client import VivekaClient` works without knowing the package name.

Do not add behaviour here. Edit `viveka/client.py` instead.
"""

from viveka.client import VivekaClient

__all__ = ["VivekaClient"]
