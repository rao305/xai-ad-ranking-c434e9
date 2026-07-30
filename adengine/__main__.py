"""Allow `python -m adengine` to run the CLI."""

from .run import main

if __name__ == "__main__":
    raise SystemExit(main())
