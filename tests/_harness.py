"""A tiny check() harness, so the tests need no third-party runner."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
_checks = 0
_failures = 0
def check(name: str, condition: bool, detail: str = "") -> None:
    global _checks, _failures
    _checks += 1
    if condition:
        print(f"  ok    {name}")
    else:
        _failures += 1
        print(f"  FAIL  {name}   {detail}")
def summary() -> int:
    print(f"\n{_checks} checks, {_failures} failures")
    return 1 if _failures else 0
