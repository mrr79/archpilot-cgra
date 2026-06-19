"""
Config Loader: builds a PEArray from user-supplied JSON instructions and
TXT initial register data.

Allows a person who is not editing Python code to define a full CGRA
simulation through two plain files plus a mesh size, then run it from
the command line.

File formats
------------

INSTRUCTIONS (.json)
    A JSON object with three top-level keys:

      "rows", "cols"        (optional ints — see CLI --rows/--cols priority
                              rules below)
      "topology"             (optional str: "mesh" or "torus", default "mesh")
      "program"               a list of cycles; each cycle is a JSON object
                              whose keys are "row,col" strings and whose
                              values are instruction objects:
                                {"opcode": "ADD"|"MUL"|"COMPLEMENT"|"NOP",
                                 "src_a": int, "src_b": int,
                                 "dst": int, "mux_sel": int}
                              A PE with no entry for a given cycle executes
                              NOP automatically.

    Example:
        {
          "rows": 2,
          "cols": 2,
          "topology": "mesh",
          "program": [
            {
              "0,0": {"opcode": "ADD", "src_a": 0, "src_b": 1,
                       "dst": 2, "mux_sel": 0}
            },
            {
              "0,0": {"opcode": "MUL", "src_a": 2, "src_b": 3,
                       "dst": 0, "mux_sel": 0}
            }
          ]
        }

INITIAL DATA (.txt)
    One register assignment per line, in the form:

        row,col,register_index,value

    Blank lines and lines starting with '#' are ignored.

    Example:
        # PE(0,0) operands
        0,0,0,10
        0,0,1,5
        0,0,3,3

Mesh size precedence
---------------------
The mesh size used for the run is resolved with this priority, highest
first:
    1. --rows / --cols command-line flags (if both given)
    2. "rows" / "cols" keys inside the JSON file (if both present)
    3. Inferred from the highest row/col index referenced anywhere in
       the JSON program or the TXT data, plus 1.

If the resolved size is smaller than an index referenced in either file,
a ConfigLoaderError is raised before the simulation starts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from archpilot_cgra import ConfigMemory, PEArray
from archpilot_cgra.exceptions import SimulationException


class ConfigLoaderError(SimulationException):
    """Raised when the user-supplied JSON/TXT files are invalid."""


def _parse_pe_key(key: str) -> tuple[int, int]:
    """Parse a 'row,col' JSON key into a (row, col) tuple."""
    try:
        row_str, col_str = key.split(",")
        return int(row_str.strip()), int(col_str.strip())
    except (ValueError, AttributeError) as exc:
        raise ConfigLoaderError(
            f"Invalid PE key {key!r}; expected format 'row,col' (e.g. '0,1')."
        ) from exc


def load_instructions_json(path: str | Path) -> dict[str, Any]:
    """
    Load and lightly validate the instructions JSON file.

    Returns the parsed dict with keys: rows (optional), cols (optional),
    topology (optional), program (required, list of cycle dicts keyed
    by (row, col) tuples instead of strings).

    Raises:
        ConfigLoaderError: on missing file, invalid JSON, missing
            "program" key, or malformed instruction entries.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigLoaderError(f"Instructions file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigLoaderError(f"Invalid JSON in {path}: {exc}") from exc

    if "program" not in raw:
        raise ConfigLoaderError(f"{path} is missing required key 'program'.")

    required_fields = {"opcode", "src_a", "src_b", "dst", "mux_sel"}
    program: list[dict[tuple[int, int], dict]] = []

    for cycle_idx, cycle_obj in enumerate(raw["program"]):
        if not isinstance(cycle_obj, dict):
            raise ConfigLoaderError(
                f"Cycle {cycle_idx} in {path} must be a JSON object."
            )
        cycle_parsed: dict[tuple[int, int], dict] = {}
        for pe_key, instr in cycle_obj.items():
            pe_id = _parse_pe_key(pe_key)
            missing = required_fields - instr.keys()
            if missing:
                raise ConfigLoaderError(
                    f"Cycle {cycle_idx}, PE {pe_key} in {path} is missing "
                    f"field(s): {sorted(missing)}."
                )
            cycle_parsed[pe_id] = {
                "opcode": instr["opcode"],
                "src_a": int(instr["src_a"]),
                "src_b": int(instr["src_b"]),
                "dst": int(instr["dst"]),
                "mux_sel": int(instr["mux_sel"]),
            }
        program.append(cycle_parsed)

    return {
        "rows": raw.get("rows"),
        "cols": raw.get("cols"),
        "topology": raw.get("topology", "mesh"),
        "program": program,
    }


def load_initial_data_txt(path: str | Path) -> list[tuple[int, int, int, int]]:
    """
    Load the initial register data TXT file.

    Each non-comment, non-blank line must have the form
    'row,col,register_index,value'.

    Returns a list of (row, col, register_index, value) tuples.

    Raises:
        ConfigLoaderError: on missing file or a malformed line.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigLoaderError(f"Initial data file not found: {path}")

    entries: list[tuple[int, int, int, int]] = []
    for line_num, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            raise ConfigLoaderError(
                f"{path}:{line_num}: expected 'row,col,register,value', "
                f"got {raw_line!r}."
            )
        try:
            row, col, reg, value = (int(p) for p in parts)
        except ValueError as exc:
            raise ConfigLoaderError(
                f"{path}:{line_num}: all four fields must be integers, "
                f"got {raw_line!r}."
            ) from exc
        entries.append((row, col, reg, value))

    return entries


def _infer_mesh_size(
    program: list[dict[tuple[int, int], dict]],
    initial_data: list[tuple[int, int, int, int]],
) -> tuple[int, int]:
    """Infer (rows, cols) from the highest row/col index referenced anywhere."""
    max_row, max_col = 0, 0
    for cycle in program:
        for (r, c) in cycle:
            max_row, max_col = max(max_row, r), max(max_col, c)
    for (r, c, _reg, _val) in initial_data:
        max_row, max_col = max(max_row, r), max(max_col, c)
    return max_row + 1, max_col + 1


def build_array_from_files(
    instructions_path: str | Path,
    initial_data_path: str | Path,
    rows: int | None = None,
    cols: int | None = None,
    rf_depth: int = 16,
    data_width: int = 32,
) -> PEArray:
    """
    Build a fully configured, ready-to-run PEArray from user files.

    Args:
        instructions_path: Path to the instructions .json file.
        initial_data_path: Path to the initial data .txt file.
        rows: Mesh rows override (takes priority over the JSON file and
              over auto-inference). Must be given together with cols.
        cols: Mesh cols override. Must be given together with rows.
        rf_depth: Register file depth for every PE (default 16).
        data_width: Bit width for overflow checking (default 32).

    Returns:
        A PEArray with the program loaded into ConfigMemory and all
        initial register values already written.

    Raises:
        ConfigLoaderError: if files are invalid, or if a referenced
            PE/register index falls outside the resolved mesh size.

    Example:
        >>> array = build_array_from_files(
        ...     "instructions.json", "initial_data.txt", rows=2, cols=2
        ... )
        >>> array.run(5)
    """
    parsed = load_instructions_json(instructions_path)
    initial_data = load_initial_data_txt(initial_data_path)

    # ---- Resolve mesh size (CLI > JSON > inferred) -----------------------
    if rows is not None and cols is not None:
        final_rows, final_cols = rows, cols
    elif parsed["rows"] is not None and parsed["cols"] is not None:
        final_rows, final_cols = parsed["rows"], parsed["cols"]
    else:
        final_rows, final_cols = _infer_mesh_size(
            parsed["program"], initial_data
        )

    # ---- Validate every reference fits inside the resolved mesh ----------
    for cycle in parsed["program"]:
        for (r, c) in cycle:
            if not (0 <= r < final_rows and 0 <= c < final_cols):
                raise ConfigLoaderError(
                    f"Instruction references PE ({r},{c}), which is "
                    f"outside the {final_rows}x{final_cols} mesh."
                )
    for (r, c, reg, _val) in initial_data:
        if not (0 <= r < final_rows and 0 <= c < final_cols):
            raise ConfigLoaderError(
                f"Initial data references PE ({r},{c}), which is "
                f"outside the {final_rows}x{final_cols} mesh."
            )
        if not (0 <= reg < rf_depth):
            raise ConfigLoaderError(
                f"Initial data references register {reg} at PE ({r},{c}), "
                f"outside rf_depth={rf_depth}."
            )

    # ---- Build the array ---------------------------------------------------
    mem = ConfigMemory(
        rows=final_rows, cols=final_cols, config_sequence=parsed["program"]
    )
    array = PEArray(
        rows=final_rows,
        cols=final_cols,
        rf_depth=rf_depth,
        data_width=data_width,
        topology=parsed["topology"],
        config_memory=mem,
    )

    for (r, c, reg, val) in initial_data:
        array.get_pe(r, c).rf.write(reg, val)

    return array