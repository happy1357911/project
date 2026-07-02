from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import Iterable

DEFAULT_PRIMARY_CHECKPOINT = "results_v74/five_runs/run_13_tiny_m15_bc_dominant_r010/no_meta_seed_0/best_model.pth"

DEFAULT_CHECKPOINT_GLOBS = (
    "results_v74/five_runs/**/best_model.pth",
    "results_v74_locked_test/five_runs/**/best_model.pth",
    "all/results_v74/five_runs/**/best_model.pth",
    "all/results_v74_locked_test/five_runs/**/best_model.pth",
)

EXCLUDED_DEFAULT_PARTS = {".codex_tmp", "backups"}


def parse_csv_list(text: str | None) -> list[str]:
    if text is None:
        return []
    return [item.strip() for item in str(text).split(",") if item and item.strip()]


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _is_excluded_default(path: Path) -> bool:
    parts = {part.lower() for part in path.parts}
    return any(part.lower() in parts for part in EXCLUDED_DEFAULT_PARTS)


def expand_checkpoint_globs(patterns: str | Iterable[str] | None, *, exclude_default_parts: bool = True) -> list[Path]:
    if patterns is None:
        return []
    if isinstance(patterns, str):
        pattern_list = parse_csv_list(patterns)
    else:
        pattern_list = [str(pattern).strip() for pattern in patterns if str(pattern).strip()]
    paths: list[Path] = []
    for pattern in pattern_list:
        matches = [Path(match) for match in glob(pattern, recursive=True)]
        for path in sorted(matches, key=lambda item: str(item)):
            if exclude_default_parts and _is_excluded_default(path):
                continue
            paths.append(path)
    return paths


def _append_existing_unique(out: list[Path], seen: set[str], path: Path, *, require_exists: bool, source: str) -> None:
    if require_exists and not path.exists():
        raise FileNotFoundError(f"Checkpoint from {source} does not exist: {path}")
    if not path.exists():
        return
    key = _path_key(path)
    if key not in seen:
        out.append(path)
        seen.add(key)


def resolve_checkpoint_paths(
    checkpoint: str | Path | None = None,
    checkpoints: str | Iterable[str] | None = None,
    checkpoint_glob: str | Iterable[str] | None = None,
    *,
    default_checkpoint: str | Path | None = DEFAULT_PRIMARY_CHECKPOINT,
    search_defaults: bool = True,
    require_exists: bool = True,
    max_count: int | None = None,
) -> list[Path]:
    """Resolve checkpoint paths consistently across root/, locked-test/, and all/ packages."""
    out: list[Path] = []
    seen: set[str] = set()
    checked_sources: list[str] = []

    explicit_values: list[str | Path] = []
    if checkpoint:
        explicit_values.append(checkpoint)
    if isinstance(checkpoints, str) or checkpoints is None:
        explicit_values.extend(parse_csv_list(checkpoints))
    else:
        explicit_values.extend([value for value in checkpoints if str(value).strip()])

    for value in explicit_values:
        checked_sources.append(str(value))
        _append_existing_unique(out, seen, Path(value), require_exists=require_exists, source="explicit argument")

    glob_matches = expand_checkpoint_globs(checkpoint_glob, exclude_default_parts=False)
    checked_sources.extend(parse_csv_list(checkpoint_glob) if isinstance(checkpoint_glob, str) else [str(p) for p in checkpoint_glob or []])
    for path in glob_matches:
        _append_existing_unique(out, seen, path, require_exists=False, source="checkpoint glob")

    if not out and search_defaults:
        if default_checkpoint:
            default_path = Path(default_checkpoint)
            checked_sources.append(str(default_path))
            if default_path.exists():
                _append_existing_unique(out, seen, default_path, require_exists=False, source="default checkpoint")
        for pattern in DEFAULT_CHECKPOINT_GLOBS:
            checked_sources.append(pattern)
            for path in expand_checkpoint_globs(pattern):
                _append_existing_unique(out, seen, path, require_exists=False, source="default glob")

    if max_count is not None:
        out = out[: max(0, int(max_count))]

    if require_exists and not out:
        checked = ", ".join(checked_sources) if checked_sources else "no checkpoint sources"
        raise FileNotFoundError(f"No checkpoint resolved. Checked: {checked}")
    return out


def resolve_single_checkpoint(**kwargs) -> str:
    paths = resolve_checkpoint_paths(max_count=1, **kwargs)
    if not paths:
        raise FileNotFoundError("No checkpoint resolved.")
    return str(paths[0].resolve())
