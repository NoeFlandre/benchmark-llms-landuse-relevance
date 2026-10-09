import json
from pathlib import Path

import pytest

from landuse_relevance_bench.adapters.g5k_sites import (
    SiteConfig,
    load_site_configs,
    parse_site_configs,
)
from landuse_relevance_bench.domain.sharding import model_language_pairs
from scripts.g5k_collect import allocate_pairs


def _site(**overrides: object) -> dict[str, object]:
    site: dict[str, object] = {"name": "nancy", "frontend": "nancy", "weight": 1}
    site.update(overrides)
    return site


def test_valid_config_parses_into_site_configs(tmp_path: Path) -> None:
    path = tmp_path / "sites.json"
    path.write_text(
        json.dumps(
            {
                "sites": [
                    _site(name="nancy", frontend="nancy", weight=2),
                    _site(name="grenoble", frontend="grenoble-fe", weight=1),
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_site_configs(path) == (
        SiteConfig("nancy", "nancy", 2),
        SiteConfig("grenoble", "grenoble-fe", 1),
    )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ([], "sites config must be a JSON object"),
        ({}, "sites config must contain a non-empty sites list"),
        ({"sites": []}, "sites config must contain a non-empty sites list"),
        ({"sites": "nancy"}, "sites config must contain a non-empty sites list"),
        ({"sites": ["nancy"]}, "each site must be an object"),
        ({"sites": [_site(name="")]}, "invalid site entry"),
        ({"sites": [_site(name=None)]}, "invalid site entry"),
        ({"sites": [_site(frontend="")]}, "invalid site entry"),
        ({"sites": [_site(frontend=None)]}, "invalid site entry"),
        ({"sites": [_site(weight=0)]}, "invalid site entry"),
        ({"sites": [_site(weight=-1)]}, "invalid site entry"),
        ({"sites": [_site(weight=1.5)]}, "invalid site entry"),
        ({"sites": [_site(weight=True)]}, "invalid site entry"),
        ({"sites": [_site(weight="2")]}, "invalid site entry"),
        ({"sites": [_site(name="  ")]}, "site names must be non-empty and unique"),
        (
            {"sites": [_site(name="nancy"), _site(name="nancy", frontend="other")]},
            "site names must be non-empty and unique",
        ),
    ],
)
def test_invalid_config_is_rejected_with_existing_message(config: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_site_configs(config)


@pytest.mark.parametrize(
    ("sites", "message"),
    [
        ((), "at least one site is required"),
        (
            (SiteConfig("nancy", "nancy", 1), SiteConfig(" ", "grenoble", 1)),
            "site names must be non-empty and unique",
        ),
        (
            (SiteConfig("nancy", "nancy", 1), SiteConfig("nancy", "grenoble", 1)),
            "site names must be non-empty and unique",
        ),
        ((SiteConfig("nancy", "nancy", 0),), "site weights must be positive"),
    ],
)
def test_allocation_rejects_invalid_sites_with_existing_message(
    sites: tuple[SiteConfig, ...], message: str
) -> None:
    pairs = model_language_pairs(["a/model"], ["en"])

    with pytest.raises(ValueError, match=message):
        allocate_pairs(pairs, sites)
