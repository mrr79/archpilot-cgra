"""
PPA estimation example for ArchPilot-CGRA.

Demonstrates the full PPA Engine workflow:
  1. Create a 4×4 PE array.
  2. Load and run a 100-cycle workload (counter on every PE).
  3. Estimate Power, Performance, and Area.
  4. Print a formatted summary.
  5. Export a CSV report (SYRS-USA-003).
  6. Render a utilization heatmap (SYRS-USA-004).

Run:
    python -m examples.ppa_estimation
    # or from the repo root:
    python examples/ppa_estimation.py
"""

from archpilot_cgra import ConfigMemory, PEArray
from archpilot_cgra.heatmap import HeatmapGenerator
from archpilot_cgra.ppa_engine import PPAEngine, TechnologyParams

ROWS = 4
COLS = 4
NUM_CYCLES = 100


def build_counter_program(rows: int, cols: int, cycles: int) -> ConfigMemory:
    """
    Program every PE to increment rf[0] by 1 each cycle (ADD rf[0], rf[1]).

    This saturates all FUs and produces a uniform utilization heatmap.
    """
    instr = {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 0, "mux_sel": 0}
    cycle_cfg = {(r, c): instr for r in range(rows) for c in range(cols)}
    return ConfigMemory(
        rows=rows,
        cols=cols,
        config_sequence=[cycle_cfg] * cycles,
    )


def main() -> None:
    print(f"Setting up {ROWS}×{COLS} mesh, {NUM_CYCLES} cycles...\n")

    # ── Simulation ───────────────────────────────────────────────────
    mem = build_counter_program(ROWS, COLS, NUM_CYCLES)
    array = PEArray(rows=ROWS, cols=COLS, config_memory=mem)

    # Initialise rf[1] = 1 on all PEs so the counter increments by 1
    for r in range(ROWS):
        for c in range(COLS):
            array.get_pe(r, c).rf.write(1, 1)

    array.run(NUM_CYCLES)

    # ── PPA Estimation ───────────────────────────────────────────────
    # Using default 28 nm parameters; can be customised:
    # tech = TechnologyParams(freq_mhz=1000.0, vdd_v=0.85)
    tech = TechnologyParams()
    engine = PPAEngine(array, tech)
    report = engine.estimate()

    print(report.summary())
    print()

    # ── CSV report ────────────────────────────────────────────────────
    csv_path = "ppa_report.csv"
    engine.to_csv(csv_path)
    print(f"CSV report written → {csv_path}")

    # ── Heatmap (ASCII) ───────────────────────────────────────────────
    summary = array.get_activity_summary()
    heatmap_str = HeatmapGenerator.ascii(
        summary, ROWS, COLS, max_cycles=NUM_CYCLES
    )
    print(heatmap_str)

    # ── Heatmap (CSV) ────────────────────────────────────────────────
    heatmap_csv = "heatmap.csv"
    HeatmapGenerator.to_csv(
        summary, ROWS, COLS, path=heatmap_csv, max_cycles=NUM_CYCLES
    )
    print(f"Heatmap CSV written → {heatmap_csv}")

    # ── Sanity check (SYRS-PER-002) ──────────────────────────────────
    print(f"\nEstimation completed in {report.elapsed_s * 1000:.2f} ms"
          f"  (limit: 5000 ms)")
    assert report.elapsed_s < 5.0, "SYRS-PER-002 violated: estimate took > 5 s"
    print("SYRS-PER-002 ✓  (estimate completed in < 5 s)")


if __name__ == "__main__":
    main()