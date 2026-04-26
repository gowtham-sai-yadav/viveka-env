"""Top-level `server.app` shim for OpenEnv canonical layout.

Real implementation lives at `viveka.server.app`. This module re-exports the
FastAPI `app` object and provides a `main()` entry point so that:

  * `openenv push --validate` finds `server/app.py` at repo root
  * `[project.scripts] server = "server.app:main"` resolves
  * `pip install git+https://huggingface.co/spaces/...` followed by
    `from server.app import app` gets the same FastAPI instance the Docker
    container serves.

Do not add behaviour here. Edit `viveka/server/app.py` instead.
"""

from viveka.server.app import app

__all__ = ["app", "main"]


def main() -> None:
    """Run the Viveka FastAPI server with uvicorn on 0.0.0.0:8000."""
    import uvicorn

    uvicorn.run("viveka.server.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
