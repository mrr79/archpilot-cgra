"""
Heatmap module for the ArchPilot-CGRA simulator.

Generates utilization heatmaps of the PE mesh after simulation.
Two output formats are supported:

  ASCII   — terminal-printable heatmap using Unicode block characters.
  CSV     — machine-readable grid with raw counts and utilization %.

Complies with: SYRS-USA-004, SYRS-MNT-001, SYRS-MNT-002, SYRS-QLY-002
"""

from __future__ import annotations

import csv
from pathlib import Path


# Unicode block characters ordered by density (light → full)
_BLOCKS: tuple[str, ...] = (" ", "░", "▒", "▓", "█")

# Bucket thresholds for the five density levels (fraction 0–1)
_THRESHOLDS: tuple[float, ...] = (0.0, 0.20, 0.45, 0.70, 0.90)


class HeatmapGenerator:
    """
    PE utilization heatmap generator (SYRS-USA-004).

    Consumes the activity_summary produced by PEArray.get_activity_summary()
    and renders a visual or CSV representation of the utilization of each
    PE in the mesh.

    Usage:
        >>> summary = array.get_activity_summary()
        >>> print(HeatmapGenerator.ascii(summary, array.rows, array.cols,
        ...                              max_cycles=100))
        >>> HeatmapGenerator.to_csv(summary, array.rows, array.cols,
        ...                          path="heatmap.csv", max_cycles=100)

    Complies with: SYRS-USA-004
    """

    @staticmethod
    def ascii(
        activity_summary: dict[str, int],
        rows: int,
        cols: int,
        max_cycles: int | None = None,
    ) -> str:
        """
        Render a terminal-printable heatmap using Unicode block characters.

        Utilization ranges map to characters as follows:
          0 %–19 %  →  ' '  (empty)
          20 %–44 % →  '░'  (light shade)
          45 %–69 % →  '▒'  (medium shade)
          70 %–89 % →  '▓'  (dark shade)
          90 %–100% →  '█'  (full block)

        Args:
            activity_summary: Mapping of str(pe_id) → active_cycle_count,
                              as returned by PEArray.get_activity_summary().
            rows:             Number of mesh rows.
            cols:             Number of mesh columns.
            max_cycles:       Total simulated cycles (denominator for
                              utilization %).  If None, uses the maximum
                              count across all PEs.

        Returns:
            A multi-line string ready to be printed to the terminal.

        Example:
            >>> summary = {"(0, 0)": 90, "(0, 1)": 10}
            >>> print(HeatmapGenerator.ascii(summary, 1, 2, max_cycles=100))
        """
        grid = HeatmapGenerator._build_grid(
            activity_summary, rows, cols, max_cycles
        )

        col_width = 6  # characters per cell including borders
        separator = "+" + (("─" * col_width + "+") * cols)

        lines: list[str] = [
            "",
            "  PE Utilization Heatmap  "
            "(░ <20%  ▒ 20-44%  ▓ 45-69%  █ ≥70%)",
            separator,
        ]

        for r in range(rows):
            cell_line = "|"
            for c in range(cols):
                util = grid[r][c]
                block = HeatmapGenerator._block(util)
                pct = f"{util * 100:.0f}%"
                cell_line += f" {block}{block} {pct:>3} |"
            lines.append(cell_line)
            lines.append(separator)

        return "\n".join(lines)

    @staticmethod
    def to_csv(
        activity_summary: dict[str, int],
        rows: int,
        cols: int,
        path: str | Path,
        max_cycles: int | None = None,
    ) -> None:
        """
        Write the utilization heatmap to a CSV file (SYRS-USA-004).

        The CSV contains one row per PE with columns:
          row, col, active_cycles, total_cycles, utilization_pct

        Args:
            activity_summary: Mapping of str(pe_id) → active_cycle_count.
            rows:             Number of mesh rows.
            cols:             Number of mesh columns.
            path:             Destination file path.
            max_cycles:       Total simulated cycles.  If None, uses max
                              count across all PEs.

        Example:
            >>> HeatmapGenerator.to_csv(summary, 2, 2, "heatmap.csv", 100)
        """
        grid = HeatmapGenerator._build_grid(
            activity_summary, rows, cols, max_cycles
        )
        max_count = (
            max_cycles
            if max_cycles is not None
            else max(activity_summary.values(), default=1)
        )
        if max_count == 0:
            max_count = 1

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["row", "col", "pe_id", "active_cycles",
                 "total_cycles", "utilization_pct"]
            )
            for r in range(rows):
                for c in range(cols):
                    pe_id = str((r, c))
                    active = activity_summary.get(pe_id, 0)
                    util_pct = round(grid[r][c] * 100, 2)
                    writer.writerow(
                        [r, c, pe_id, active, max_count, util_pct]
                    )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_grid(
        activity_summary: dict[str, int],
        rows: int,
        cols: int,
        max_cycles: int | None,
    ) -> list[list[float]]:
        """Build a 2D grid of utilization fractions in [0.0, 1.0]."""
        if max_cycles is None:
            max_cycles = max(activity_summary.values(), default=1)
        if max_cycles == 0:
            max_cycles = 1

        return [
            [
                activity_summary.get(str((r, c)), 0) / max_cycles
                for c in range(cols)
            ]
            for r in range(rows)
        ]

    @staticmethod
    def _block(utilization: float) -> str:
        """Map a utilization fraction [0, 1] to a Unicode block character."""
        for threshold, block in zip(
            reversed(_THRESHOLDS), reversed(_BLOCKS)
        ):
            if utilization >= threshold:
                return block
        return _BLOCKS[0]