from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


class ConfigError(RuntimeError):
    pass


GENERIC_CONFIG_KEYS = {
    "fit_noise_type",
    "fit_noise_scale",
    "fit_noise_mode",
    "fit_noise_seed",
    "fit_noise_clip_nonnegative",
    "fit_noise_preserve_initial",
}


def add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a SysBioSR YAML config file.",
    )


def _coerce_value(action: argparse.Action, value: Any) -> Any:
    if value is None:
        return None
    if isinstance(action, argparse._StoreTrueAction) or isinstance(action, argparse._StoreFalseAction):
        return bool(value)
    if getattr(action, "type", None) is not None:
        try:
            return action.type(value)
        except Exception as exc:
            raise ConfigError(f"Failed to cast config value for '{action.dest}': {value!r}") from exc
    return value


def _flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    # allow either a flat dict or a dict with a top-level `params` block
    if "params" in config and isinstance(config["params"], dict):
        merged = dict(config)
        params = merged.pop("params")
        merged.update(params)
        return merged
    return config


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_inherits(value: Any, *, config_path: Path) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        raise ConfigError(f"'inherits' must be a path or a list of paths: {config_path}")

    resolved: list[Path] = []
    for item in items:
        path = Path(item).expanduser()
        if not path.is_absolute():
            path = config_path.parent / path
        resolved.append(path.resolve())
    return resolved


def _load_yaml_config_recursive(
    config_path: Path,
    *,
    stack: tuple[Path, ...],
) -> tuple[dict[str, Any], dict[str, Path]]:
    if config_path in stack:
        cycle = " -> ".join(str(p) for p in (*stack, config_path))
        raise ConfigError(f"Config inheritance cycle detected: {cycle}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if not isinstance(config, dict):
        raise ConfigError(f"Config root must be a mapping/dict: {config_path}")

    config = _flatten_config(config)
    parent_paths = _normalize_inherits(config.pop("inherits", None), config_path=config_path)

    merged: dict[str, Any] = {}
    sources: dict[str, Path] = {}
    next_stack = (*stack, config_path)
    for parent_path in parent_paths:
        parent_config, parent_sources = _load_yaml_config_recursive(parent_path, stack=next_stack)
        merged = _merge_dicts(merged, parent_config)
        sources.update(parent_sources)

    merged = _merge_dicts(merged, config)
    for key in config:
        sources[key] = config_path
    return merged, sources


def load_yaml_config_with_sources(config_path: str | Path) -> tuple[Path, dict[str, Any], dict[str, Path]]:
    config_path = Path(config_path).expanduser().resolve()
    config, sources = _load_yaml_config_recursive(config_path, stack=())
    return config_path, config, sources


def load_yaml_config(config_path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path, config, _ = load_yaml_config_with_sources(config_path)
    return config_path, config


def validate_config_keys(
    config: dict[str, Any],
    *,
    valid_keys: set[str],
    config_path: str | Path,
) -> None:
    unknown = sorted(set(config) - (set(valid_keys) | GENERIC_CONFIG_KEYS))
    if unknown:
        raise ConfigError(f"Unknown config key(s) in {Path(config_path).resolve()}: {unknown}")


def require_config_value(value: Any, *, key: str, config_path: str | Path) -> Any:
    if value is None:
        raise ConfigError(f"Missing required key '{key}' in {Path(config_path).resolve()}")
    return value


def resolve_config_key_path(
    config: dict[str, Any],
    sources: dict[str, Path],
    key: str,
    *,
    fallback_config_path: str | Path,
) -> Path | None:
    return resolve_path_from_config(
        config.get(key),
        config_path=sources.get(key, Path(fallback_config_path).resolve()),
    )


def resolve_path_from_config(value: str | Path | None, *, config_path: str | Path) -> Path | None:
    if value is None:
        return None

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(config_path).resolve().parent / path
    return path.resolve()


def parse_args_with_config(parser: argparse.ArgumentParser) -> argparse.Namespace:
    preliminary_args, _ = parser.parse_known_args()
    config_path: Path | None = getattr(preliminary_args, "config", None)

    if config_path is not None:
        config_path, config = load_yaml_config(config_path)

        valid_dests = {action.dest: action for action in parser._actions if action.dest != "help"}
        defaults: dict[str, Any] = {}
        for key, value in config.items():
            if key not in valid_dests:
                raise ConfigError(f"Unknown config key '{key}' in {config_path}")
            defaults[key] = _coerce_value(valid_dests[key], value)
        parser.set_defaults(**defaults)

    return parser.parse_args()
