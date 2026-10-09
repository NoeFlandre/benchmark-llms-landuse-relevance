"""Load and validate the Grid'5000 site list used by the multisite sharding scripts."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class SiteConfig:
    """A Grid'5000 site, its frontend host, and its relative number of shard slots."""

    name: str
    frontend: str
    weight: int


def load_site_configs(sites_config: Path) -> tuple[SiteConfig, ...]:
    """Read and validate a sites JSON file."""
    return parse_site_configs(json.loads(sites_config.read_text(encoding="utf-8")))


def parse_site_configs(config: object) -> tuple[SiteConfig, ...]:
    """Validate a decoded sites JSON object."""
    if not isinstance(config, dict):
        raise ValueError("sites config must be a JSON object")
    sites = config.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("sites config must contain a non-empty sites list")
    parsed: list[SiteConfig] = []
    for site in sites:
        if not isinstance(site, dict):
            raise ValueError("each site must be an object")
        name = site.get("name")
        frontend = site.get("frontend")
        weight = site.get("weight")
        if not isinstance(name, str) or not name:
            raise ValueError(f"invalid site entry: {site!r}")
        if not isinstance(frontend, str) or not frontend:
            raise ValueError(f"invalid site entry: {site!r}")
        if not isinstance(weight, int) or isinstance(weight, bool) or weight < 1:
            raise ValueError(f"invalid site entry: {site!r}")
        parsed.append(SiteConfig(name=name, frontend=frontend, weight=weight))
    return validate_site_configs(parsed)


def validate_site_configs(sites: Sequence[SiteConfig]) -> tuple[SiteConfig, ...]:
    """Check the rules that apply to a set of sites, whatever their source."""
    if not sites:
        raise ValueError("at least one site is required")
    names = [site.name for site in sites]
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise ValueError("site names must be non-empty and unique")
    if any(site.weight < 1 for site in sites):
        raise ValueError("site weights must be positive")
    return tuple(sites)
