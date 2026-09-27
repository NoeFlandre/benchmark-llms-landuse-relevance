"""Report complete, queued, running, failed, and unclaimed model-language pairs."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from landuse_relevance_bench.adapters.translations import load_manifest
from landuse_relevance_bench.domain.roster import model_ids
from landuse_relevance_bench.domain.sharding import model_language_pairs, shard_pairs

STATUS_COLUMNS = 3
SSH_EXECUTABLE = shutil.which("ssh")


@dataclass(frozen=True)
class PairStatus:
    site: str
    model_id: str
    language: str
    state: str
    job_id: str = "-"


@dataclass(frozen=True)
class StatusPlan:
    remote_root: str
    remote_results: str
    shard_count: int
    pairs: tuple[tuple[str, str], ...]


def _ssh(frontend: str, remote_command: str) -> str:
    if SSH_EXECUTABLE is None:
        raise RuntimeError("ssh executable is unavailable")
    result = subprocess.run(  # noqa: S603 -- frontend and remote commands are validated.
        [SSH_EXECUTABLE, "-o", "ConnectTimeout=60", frontend, remote_command],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"ssh {frontend} failed ({result.returncode}): {details}")
    return result.stdout


def _jobs_for_shards(  # noqa: C901, PLR0912
    payload: object,
    *,
    remote_results: str,
    shard_indices: set[int],
    shard_count: int,
) -> dict[int, tuple[str, str]]:
    if isinstance(payload, dict):
        jobs = payload.values()
    elif isinstance(payload, list):
        jobs = payload
    else:
        return {}
    active: dict[int, tuple[str, str]] = {}
    for job in jobs:
        if not isinstance(job, dict):
            continue
        command = str(job.get("command") or "")
        if not command:
            continue
        environment: dict[str, str] = {}
        for token in shlex.split(command):
            key, separator, value = token.partition("=")
            if separator:
                environment[key] = value
        if environment.get("LRB_RESULTS") != remote_results:
            continue
        try:
            shard_index = int(environment["LRB_SHARD_INDEX"])
            requested_shards = int(environment["LRB_SHARD_COUNT"])
        except (KeyError, ValueError):
            continue
        if shard_index not in shard_indices or requested_shards != shard_count:
            continue
        state = str(job.get("state", ""))
        if state in {"Running", "Launching", "toLaunch", "Suspended", "Resuming", "Finishing"}:
            classification = "running"
        elif state in {"Waiting", "Hold", "toAckReservation"}:
            classification = "queued"
        elif state in {"Error", "toError"}:
            classification = "failed"
        else:
            continue
        previous = active.get(shard_index)
        priority = {"queued": 1, "failed": 2, "running": 3}
        if previous is None or priority[classification] > priority[previous[0]]:
            active[shard_index] = (classification, str(job.get("id", "-")))
    return active


def _load_sites(sites_config: Path) -> list[dict[str, Any]]:
    config = json.loads(sites_config.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("sites config must be a JSON object")
    sites = config.get("sites")
    if not isinstance(sites, list) or not sites:
        raise ValueError("sites config must contain a non-empty sites list")
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
    return sites


def _status_command(plan: StatusPlan, shard_index: int) -> str:
    return (
        f"cd {shlex.quote(plan.remote_root)} && uv run --no-sync lrb status "
        f"--results-dir {shlex.quote(plan.remote_results)} "
        f"--shard-index {shard_index} --shard-count {plan.shard_count}"
    )


def _parse_status_lines(
    lines: list[str], *, site_name: str, expected: set[tuple[str, str]]
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    completed: set[tuple[str, str]] = set()
    pending: set[tuple[str, str]] = set()
    for line in lines:
        columns = line.split("\t")
        if len(columns) != STATUS_COLUMNS:
            raise ValueError(f"unexpected lrb status row from {site_name}: {line!r}")
        pair = (columns[0], columns[1])
        if columns[2] == "complete":
            completed.add(pair)
        elif columns[2] == "pending":
            pending.add(pair)
        else:
            raise ValueError(f"unexpected lrb status state from {site_name}: {line!r}")
    if completed | pending != expected or completed & pending:
        raise ValueError(f"status coverage from {site_name} does not match its plan")
    return completed, pending


def _rows_for_site_shards(
    site: dict[str, Any], shard_indices: set[int], plan: StatusPlan
) -> list[PairStatus]:
    frontend = site["frontend"]
    site_name = site["name"]
    oar_payload = json.loads(_ssh(frontend, "oarstat -u -J"))
    active = _jobs_for_shards(
        oar_payload,
        remote_results=plan.remote_results,
        shard_indices=shard_indices,
        shard_count=plan.shard_count,
    )
    rows: list[PairStatus] = []
    for shard_index in sorted(shard_indices):
        assigned_pairs = shard_pairs(plan.pairs, shard_index, plan.shard_count)
        status_lines = _ssh(frontend, _status_command(plan, shard_index)).splitlines()
        completed, _pending = _parse_status_lines(
            status_lines, site_name=site_name, expected=set(assigned_pairs)
        )
        active_state, job_id = active.get(shard_index, ("unclaimed", "-"))
        rows.extend(
            PairStatus(
                site_name,
                model_id,
                language,
                "complete" if (model_id, language) in completed else active_state,
                job_id,
            )
            for model_id, language in assigned_pairs
        )
    return rows


def _validate_pair_coverage(rows: list[PairStatus], pairs: tuple[tuple[str, str], ...]) -> None:
    seen: set[tuple[str, str]] = set()
    for row in rows:
        pair = (row.model_id, row.language)
        if pair in seen:
            raise ValueError(f"model-language pair assigned to multiple sites: {pair}")
        seen.add(pair)
    if seen != set(pairs):
        raise ValueError("multi-site assignment does not cover the complete model-language matrix")


def multisite_status(
    sites_config: Path,
    *,
    remote_root: str,
    remote_results: str,
    models: tuple[str, ...] | None = None,
    languages: tuple[str, ...] | None = None,
) -> list[PairStatus]:
    """Query each configured site and classify every pair in its fixed shard slots."""
    sites = _load_sites(sites_config)
    total_shards = sum(site["weight"] for site in sites)
    all_models = models or model_ids()
    all_languages = languages or load_manifest(Path("data/translations")).languages
    all_pairs = model_language_pairs(all_models, all_languages)
    plan = StatusPlan(remote_root, remote_results, total_shards, all_pairs)
    rows: list[PairStatus] = []
    shard_offset = 0

    for site in sites:
        shard_indices = set(range(shard_offset, shard_offset + site["weight"]))
        rows.extend(_rows_for_site_shards(site, shard_indices, plan))
        shard_offset += site["weight"]

    _validate_pair_coverage(rows, all_pairs)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites-config", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--results-dir", default="results-live")
    arguments = parser.parse_args()
    remote_results = arguments.results_dir
    if not remote_results.startswith("/"):
        remote_results = f"{arguments.remote_root.rstrip('/')}/{remote_results}"
    try:
        rows = multisite_status(
            arguments.sites_config,
            remote_root=arguments.remote_root,
            remote_results=remote_results,
        )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print("site\tmodel\tlanguage\tstate\tjob_id")
    for row in rows:
        print(f"{row.site}\t{row.model_id}\t{row.language}\t{row.state}\t{row.job_id}")
    counts = Counter(row.state for row in rows)
    print(
        "\n"
        + ", ".join(
            f"{state}={counts.get(state, 0)}"
            for state in ("complete", "running", "queued", "failed", "unclaimed")
        )
    )


if __name__ == "__main__":
    main()
