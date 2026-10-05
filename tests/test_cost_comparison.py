"""Locks the README's published cost table to the code that computes it."""
import json
from pathlib import Path

import pytest

from cost_comparison import api_cost_usd, project_to_scale, rented_compute_cost_usd

ROOT = Path(__file__).parent.parent
PUBLISHED_TOKENS = 13_111  # README section 1; not stored in the committed JSON


@pytest.fixture(scope="module")
def warm():
    return json.loads((ROOT / "results_local_warm.json").read_text())


def test_published_run_matches_readme(warm):
    assert warm["corpus"]["num_chunks"] == 38
    assert warm["corpus"]["total_words"] == 8091
    assert warm["benchmark"]["embedding_dim"] == 384
    assert round(warm["benchmark"]["documents_per_second"], 2) == 4.26


def test_published_costs_match_readme(warm):
    wall = warm["benchmark"]["wall_clock_seconds"]
    assert api_cost_usd(PUBLISHED_TOKENS) == pytest.approx(0.000262, abs=5e-7)
    assert rented_compute_cost_usd(wall) == pytest.approx(0.000360, abs=5e-7)


def test_published_projection_to_one_million_chunks(warm):
    proj = project_to_scale(38, PUBLISHED_TOKENS, warm["benchmark"]["documents_per_second"], 1_000_000)
    assert proj["source"] == "projected"
    assert round(proj["projected_api_cost_usd"], 2) == 6.90
    assert round(proj["projected_local_rented_cost_usd"], 2) == 9.46


def test_projection_scales_linearly():
    a = project_to_scale(10, 1000, 2.0, 100)
    b = project_to_scale(10, 1000, 2.0, 200)
    assert b["projected_tokens"] == pytest.approx(2 * a["projected_tokens"])
    assert b["projected_local_seconds"] == pytest.approx(2 * a["projected_local_seconds"])
