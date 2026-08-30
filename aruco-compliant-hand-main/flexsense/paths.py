"""Find the files that ship with the project, whatever directory you are in.

Every default in the CLI is written as a short relative path - `config/hand.yaml`,
`calibration/camera_intrinsics.json`. Those only resolve if you happen to be
standing in the source tree, which makes the installed `flexsense` command work
in exactly one directory and fail everywhere else with a confusing error.

Reads go through `resolve()`: the current directory wins, so a local edit always
beats the shipped copy, and the project's own files are the fallback. Writes are
left alone - output belongs where the user is, not next to the source.
"""

from __future__ import annotations

from pathlib import Path

MARKERS = ("pyproject.toml", "config", "flexsense")


def project_root() -> Path | None:
    """The directory holding this project, or None once installed as a wheel."""
    for parent in Path(__file__).resolve().parents:
        if all((parent / marker).exists() for marker in MARKERS):
            return parent
    return None


def bundled_root() -> Path | None:
    """Data copied inside the package, for a non-editable install."""
    candidate = Path(__file__).resolve().parent / "_bundled"
    return candidate if candidate.is_dir() else None


def candidates(relative: str | Path) -> list[Path]:
    relative = Path(relative)
    if relative.is_absolute():
        return [relative]
    found = [Path.cwd() / relative]
    for root in (project_root(), bundled_root()):
        if root is not None:
            found.append(root / relative)
    return found


def resolve(relative: str | Path) -> Path:
    """First existing candidate, or the cwd-relative path so errors name it."""
    options = candidates(relative)
    for option in options:
        if option.exists():
            return option
    return options[0]


def describe(path: str | Path) -> str:
    """A short, honest label for where a file was actually picked up from."""
    resolved = Path(path).resolve()
    root = project_root()
    if root is not None:
        try:
            return f"{resolved.relative_to(root)} (project)"
        except ValueError:
            pass
    try:
        return f"{resolved.relative_to(Path.cwd())} (here)"
    except ValueError:
        return str(resolved)
