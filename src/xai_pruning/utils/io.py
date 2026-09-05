"""Path and JSON helpers without experiment-specific policy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def first_existing(candidates: Iterable[str | Path | None], description: str) -> Path:
    """Return the first existing candidate path or report every attempted path."""

    attempted = []
    for candidate in candidates:
        if candidate is None:
            continue
        path = Path(candidate).expanduser()
        attempted.append(path)
        if path.exists():
            return path.resolve()
    rendered = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(f"Could not locate {description}. Tried:\n{rendered}")


def load_json(path: str | Path) -> Any:
    """Read UTF-8 JSON."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(value: Any, path: str | Path) -> Path:
    """Write indented UTF-8 JSON, creating only the requested parent directory."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    return output_path
