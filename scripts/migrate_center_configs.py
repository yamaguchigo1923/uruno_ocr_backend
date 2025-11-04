from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict

# Allow running from anywhere
BASE = Path(__file__).resolve().parents[1]
CENTERS_DIR = BASE / "config" / "centers"

# Make 'backend' importable
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

# Import defaults from matching engine
try:
    from services.matching_module.engine import DEFAULT_CONFIGS as ENGINE_DEFAULTS
except Exception as exc:  # pragma: no cover - simple script
    raise SystemExit(f"Failed to import engine defaults: {exc}")


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: Dict[str, Any]) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")


def migrate() -> int:
    if not CENTERS_DIR.exists():
        print(f"Centers dir not found: {CENTERS_DIR}")
        return 2
    updated = 0
    for path in sorted(CENTERS_DIR.glob("*.json")):
        data = _load_json(path)
        if not data:
            continue
        changed = False
        # Remove 'headers' if present
        if "headers" in data:
            data.pop("headers", None)
            changed = True
        # Ensure nutrition/sample blocks
        for key in ("nutrition", "sample"):
            desired = ENGINE_DEFAULTS.get(key)
            if not isinstance(desired, dict):
                continue
            if data.get(key) != desired:
                data[key] = desired
                changed = True
        if changed:
            _save_json(path, data)
            updated += 1
            print(f"[UPDATED] {path.name}")
    print(f"Done. Updated {updated} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(migrate())
