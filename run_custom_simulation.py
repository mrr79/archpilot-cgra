#!/usr/bin/env python3
"""
Run a custom CGRA simulation from user-supplied instruction and data files.

This script lets a user define an entire simulation WITHOUT writing any
Python code:

  - instructions.json  -> what each PE does, cycle by cycle
  - initial_data.txt   -> starting register values for each PE
  - --rows / --cols     -> mesh size (optional; can also come from the
                            JSON file, or be auto-detected)

Usage
-----
    python3 run_custom_simulation.py \\
        --instructions instructions.json \\
        --data initial_data.txt \\
        --rows 2 --cols 2 \\
        --cycles 5

    # Mesh size taken from the JSON file or auto-inferred:
    python3 run_custom_simulation.py \\
        --instructions instructions.json \\
        --data initial_data.txt

Run with --help for the full list of options.
"""

from __future__ import annotations

import argparse
import sys

from archpilot_cgra.config_loader import ConfigLoaderError, build_array_from_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a CGRA simulation defined by JSON instructions "
                    "and a TXT file of initial register values.",
    )
    parser.add_argument(
        "--instructions", "-i", required=True,
        help="Path to the instructions .json file.",
    )
    parser.add_argument(
        "--data", "-d", required=True,
        help="Path to the initial data .txt file.",
    )
    parser.add_argument(
        "--rows", type=int, default=None,
        help="Mesh rows. Must be given together with --cols. "
             "Overrides any size in the JSON file.",
    )
    parser.add_argument(
        "--cols", type=int, default=None,
        help="Mesh cols. Must be given together with --rows.",
    )
    parser.add_argument(
        "--cycles", type=int, default=None,
        help="Number of cycles to run. Defaults to the number of cycles "
             "defined in the instructions file.",
    )
    parser.add_argument(
        "--rf-depth", type=int, default=16,
        help="Register file depth per PE (2-16, default 16).",
    )
    parser.add_argument(
        "--data-width", type=int, default=32,
        help="Bit width used for overflow checking (default 32).",
    )
    parser.add_argument(
        "--show-registers", action="store_true",
        help="Print the full register file of every PE after the run.",
    )

    args = parser.parse_args(argv)
    if (args.rows is None) != (args.cols is None):
        parser.error("--rows and --cols must be given together.")
    return args


def _result_value(results: dict, pe_id: tuple[int, int]) -> int:
    """Read a result regardless of whether step() keys by tuple or str."""
    if pe_id in results:
        return results[pe_id]
    return results[str(pe_id)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        array = build_array_from_files(
            instructions_path=args.instructions,
            initial_data_path=args.data,
            rows=args.rows,
            cols=args.cols,
            rf_depth=args.rf_depth,
            data_width=args.data_width,
        )
    except ConfigLoaderError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 1

    num_cycles = (
        args.cycles if args.cycles is not None else array.config_memory.num_cycles
    )
    if num_cycles < 1:
        print(
            "Error: no cycles to run. Provide --cycles or a non-empty "
            "'program' in the instructions file.",
            file=sys.stderr,
        )
        return 1

    # rf_depth/data_width are not always stored on PEArray itself across
    # versions of the simulator, so fall back to the values the user
    # passed in (which is also what was used to build the array).
    rf_depth = getattr(array, "rf_depth", args.rf_depth)
    data_width = getattr(array, "data_width", args.data_width)

    print(f"Mesh: {array.rows}x{array.cols}  |  Topology: {array.get_topology()}  "
          f"|  RF depth: {rf_depth}  |  Data width: {data_width}-bit")
    print(f"Running {num_cycles} cycle(s)...\n")

    try:
        history = array.run(num_cycles)
    except Exception as exc:  # noqa: BLE001 - surface any simulator exception cleanly
        print(f"Simulation error: {exc}", file=sys.stderr)
        return 1

    for record in history:
        print(f"Cycle {record['cycle']}:")
        results = record["results"]
        pe_ids = sorted(
            {pe_id for r in range(array.rows) for pe_id in [(r, c) for c in range(array.cols)]}
        )
        for pe_id in pe_ids:
            output = _result_value(results, pe_id)
            print(f"  PE{pe_id} output = {output}")
        print()

    print("Activity summary (active cycles per PE):")
    for r in range(array.rows):
        for c in range(array.cols):
            pe = array.get_pe(r, c)
            print(f"  PE({r}, {c}): {pe.activity_count} active cycle(s)")

    if args.show_registers:
        print("\nFinal register state per PE:")
        for r in range(array.rows):
            for c in range(array.cols):
                state = array.get_pe(r, c).rf.get_state()
                print(f"  PE({r}, {c}): {state}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())