"""
Unit tests for the HeatmapGenerator module (ACT-19).

Verifies ASCII heatmap rendering, CSV export, utilization bucketing,
and edge cases such as empty summaries and zero max_cycles.

Complies with: SYRS-USA-004, SYRS-REL-001
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path

import pytest

from archpilot_cgra.heatmap import HeatmapGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _summary(rows: int, cols: int, count: int) -> dict[str, int]:
    """Build a uniform activity summary with every PE at `count`."""
    return {str((r, c)): count for r in range(rows) for c in range(cols)}


# ---------------------------------------------------------------------------
# ASCII heatmap
# ---------------------------------------------------------------------------

class TestHeatmapASCII:
    """Tests for HeatmapGenerator.ascii()."""

    def test_returns_string(self) -> None:
        summary = _summary(2, 2, 50)
        result = HeatmapGenerator.ascii(summary, 2, 2, max_cycles=100)
        assert isinstance(result, str)

    def test_contains_grid_borders(self) -> None:
        summary = _summary(2, 2, 50)
        result = HeatmapGenerator.ascii(summary, 2, 2, max_cycles=100)
        assert "+" in result
        assert "─" in result

    def test_full_utilization_shows_full_block(self) -> None:
        summary = _summary(2, 2, 100)
        result = HeatmapGenerator.ascii(summary, 2, 2, max_cycles=100)
        assert "█" in result

    def test_zero_utilization_shows_empty(self) -> None:
        summary = _summary(2, 2, 0)
        result = HeatmapGenerator.ascii(summary, 2, 2, max_cycles=100)
        assert "0%" in result

    def test_contains_utilization_percentage(self) -> None:
        summary = _summary(1, 1, 100)
        result = HeatmapGenerator.ascii(summary, 1, 1, max_cycles=100)
        assert "100%" in result

    def test_correct_number_of_rows(self) -> None:
        summary = _summary(3, 2, 50)
        result = HeatmapGenerator.ascii(summary, 3, 2, max_cycles=100)
        # 3 data rows + 4 separators = 7 grid lines minimum
        lines = [l for l in result.split("\n") if "|" in l]
        assert len(lines) == 3

    def test_legend_in_output(self) -> None:
        summary = _summary(1, 1, 0)
        result = HeatmapGenerator.ascii(summary, 1, 1, max_cycles=100)
        assert "Heatmap" in result

    def test_auto_max_cycles_when_none(self) -> None:
        # max_cycles=None → uses max count as denominator
        summary = _summary(2, 2, 80)
        result = HeatmapGenerator.ascii(summary, 2, 2, max_cycles=None)
        assert "100%" in result  # 80/80 = 100%

    def test_partial_utilization_bucket(self) -> None:
        # 30 out of 100 → 30% → should show ░ (light shade, 20-44%)
        summary = {str((0, 0)): 30}
        result = HeatmapGenerator.ascii(summary, 1, 1, max_cycles=100)
        assert "░" in result or "30%" in result

    def test_1x1_mesh(self) -> None:
        summary = {str((0, 0)): 75}
        result = HeatmapGenerator.ascii(summary, 1, 1, max_cycles=100)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_4x4_mesh(self) -> None:
        summary = _summary(4, 4, 50)
        result = HeatmapGenerator.ascii(summary, 4, 4, max_cycles=100)
        lines = [l for l in result.split("\n") if "|" in l]
        assert len(lines) == 4


# ---------------------------------------------------------------------------
# CSV heatmap
# ---------------------------------------------------------------------------

class TestHeatmapCSV:
    """Tests for HeatmapGenerator.to_csv()."""

    def _run_and_read(
        self,
        summary: dict[str, int],
        rows: int,
        cols: int,
        max_cycles: int | None = 100,
    ) -> list[dict]:
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        HeatmapGenerator.to_csv(summary, rows, cols, path, max_cycles)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows_data = list(reader)
        Path(path).unlink()
        return rows_data

    def test_csv_file_created(self) -> None:
        summary = _summary(2, 2, 50)
        with tempfile.NamedTemporaryFile(
            suffix=".csv", delete=False, mode="w"
        ) as f:
            path = f.name
        HeatmapGenerator.to_csv(summary, 2, 2, path, max_cycles=100)
        assert Path(path).exists()
        Path(path).unlink()

    def test_csv_row_count(self) -> None:
        summary = _summary(2, 2, 50)
        rows_data = self._run_and_read(summary, 2, 2)
        assert len(rows_data) == 4  # 2×2 = 4 PEs

    def test_csv_4x4_row_count(self) -> None:
        summary = _summary(4, 4, 50)
        rows_data = self._run_and_read(summary, 4, 4)
        assert len(rows_data) == 16

    def test_csv_has_required_columns(self) -> None:
        summary = _summary(2, 2, 50)
        rows_data = self._run_and_read(summary, 2, 2)
        assert "row" in rows_data[0]
        assert "col" in rows_data[0]
        assert "active_cycles" in rows_data[0]
        assert "total_cycles" in rows_data[0]
        assert "utilization_pct" in rows_data[0]

    def test_full_utilization_pct(self) -> None:
        summary = _summary(1, 1, 100)
        rows_data = self._run_and_read(summary, 1, 1, max_cycles=100)
        assert float(rows_data[0]["utilization_pct"]) == 100.0

    def test_zero_utilization_pct(self) -> None:
        summary = _summary(1, 1, 0)
        rows_data = self._run_and_read(summary, 1, 1, max_cycles=100)
        assert float(rows_data[0]["utilization_pct"]) == 0.0

    def test_partial_utilization_pct(self) -> None:
        summary = {str((0, 0)): 25}
        rows_data = self._run_and_read(summary, 1, 1, max_cycles=100)
        assert float(rows_data[0]["utilization_pct"]) == 25.0

    def test_pe_id_column(self) -> None:
        summary = {str((0, 0)): 50, str((0, 1)): 50}
        rows_data = self._run_and_read(summary, 1, 2, max_cycles=100)
        pe_ids = [r["pe_id"] for r in rows_data]
        assert "(0, 0)" in pe_ids
        assert "(0, 1)" in pe_ids

    def test_missing_pe_in_summary_defaults_to_zero(self) -> None:
        # PE(0,1) not in summary → should appear with 0 active cycles
        summary = {str((0, 0)): 50}
        rows_data = self._run_and_read(summary, 1, 2, max_cycles=100)
        pe_01 = next(r for r in rows_data if r["pe_id"] == "(0, 1)")
        assert int(pe_01["active_cycles"]) == 0

    def test_parent_dir_created(self) -> None:
        summary = _summary(1, 1, 50)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "heatmap.csv"
            HeatmapGenerator.to_csv(summary, 1, 1, str(path), 100)
            assert path.exists()

    def test_auto_max_cycles(self) -> None:
        summary = {str((0, 0)): 40}
        rows_data = self._run_and_read(summary, 1, 1, max_cycles=None)
        # max_cycles=None → max count = 40 → 40/40 = 100%
        assert float(rows_data[0]["utilization_pct"]) == 100.0