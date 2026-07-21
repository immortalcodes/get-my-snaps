"""Enable ``python -m get_my_snaps`` as an alternative to the console script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
