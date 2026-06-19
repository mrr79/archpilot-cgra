"""
Tests for archpilot_cgra.config_loader (user-supplied JSON/TXT simulations).

Covers: JSON parsing and validation, TXT parsing and validation, mesh
size resolution priority (CLI > JSON > inferred), and out-of-bounds
detection for both files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archpilot_cgra.config_loader import (
    ConfigLoaderError,
    build_array_from_files,
    load_initial_data_txt,
    load_instructions_json,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


VALID_INSTR = {
    "rows": 2, "cols": 2, "topology": "mesh",
    "program": [
        {"0,0": {"opcode": "ADD", "src_a": 0, "src_b": 1,
                 "dst": 2, "mux_sel": 0}}
    ],
}

VALID_DATA = "0,0,0,10\n0,0,1,5\n"


# ---------------------------------------------------------------------------
# load_instructions_json
# ---------------------------------------------------------------------------

class TestLoadInstructionsJSON:
    def test_valid_file_parses(self, tmp_path: Path) -> None:
        p = write(tmp_path, "i.json", json.dumps(VALID_INSTR))
        parsed = load_instructions_json(p)
        assert parsed["rows"] == 2
        assert parsed["cols"] == 2
        assert parsed["topology"] == "mesh"
        assert (0, 0) in parsed["program"][0]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoaderError, match="not found"):
            load_instructions_json(tmp_path / "missing.json")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        p = write(tmp_path, "bad.json", "{not valid json")
        with pytest.raises(ConfigLoaderError, match="Invalid JSON"):
            load_instructions_json(p)

    def test_missing_program_key_raises(self, tmp_path: Path) -> None:
        p = write(tmp_path, "noprogram.json", json.dumps({"rows": 1, "cols": 1}))
        with pytest.raises(ConfigLoaderError, match="program"):
            load_instructions_json(p)

    def test_missing_instruction_field_raises(self, tmp_path: Path) -> None:
        bad = {"program": [{"0,0": {"opcode": "ADD", "src_a": 0}}]}
        p = write(tmp_path, "incomplete.json", json.dumps(bad))
        with pytest.raises(ConfigLoaderError, match="missing field"):
            load_instructions_json(p)

    def test_invalid_pe_key_raises(self, tmp_path: Path) -> None:
        bad = {"program": [{"not_a_key": {"opcode": "ADD", "src_a": 0,
                                            "src_b": 0, "dst": 0, "mux_sel": 0}}]}
        p = write(tmp_path, "badkey.json", json.dumps(bad))
        with pytest.raises(ConfigLoaderError, match="Invalid PE key"):
            load_instructions_json(p)

    def test_rows_cols_optional(self, tmp_path: Path) -> None:
        no_size = {"program": [{"0,0": {"opcode": "NOP", "src_a": 0,
                                          "src_b": 0, "dst": 0, "mux_sel": 0}}]}
        p = write(tmp_path, "nosize.json", json.dumps(no_size))
        parsed = load_instructions_json(p)
        assert parsed["rows"] is None
        assert parsed["cols"] is None

    def test_topology_defaults_to_mesh(self, tmp_path: Path) -> None:
        no_topo = {"program": []}
        p = write(tmp_path, "notopo.json", json.dumps(no_topo))
        parsed = load_instructions_json(p)
        assert parsed["topology"] == "mesh"

    def test_empty_program_is_valid(self, tmp_path: Path) -> None:
        p = write(tmp_path, "empty.json", json.dumps({"program": []}))
        parsed = load_instructions_json(p)
        assert parsed["program"] == []

    def test_multiple_pes_same_cycle(self, tmp_path: Path) -> None:
        multi = {"program": [{
            "0,0": {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 0, "mux_sel": 0},
            "0,1": {"opcode": "MUL", "src_a": 0, "src_b": 1, "dst": 0, "mux_sel": 0},
        }]}
        p = write(tmp_path, "multi.json", json.dumps(multi))
        parsed = load_instructions_json(p)
        assert len(parsed["program"][0]) == 2

    def test_field_values_cast_to_int(self, tmp_path: Path) -> None:
        # JSON numbers may come as float-looking strings in edge cases;
        # ensure ints are produced regardless.
        instr = {"program": [{"0,0": {"opcode": "ADD", "src_a": "0",
                                        "src_b": "1", "dst": "2", "mux_sel": "0"}}]}
        p = write(tmp_path, "strnums.json", json.dumps(instr))
        parsed = load_instructions_json(p)
        entry = parsed["program"][0][(0, 0)]
        assert isinstance(entry["src_a"], int)
        assert entry["src_a"] == 0


# ---------------------------------------------------------------------------
# load_initial_data_txt
# ---------------------------------------------------------------------------

class TestLoadInitialDataTXT:
    def test_valid_file_parses(self, tmp_path: Path) -> None:
        p = write(tmp_path, "d.txt", VALID_DATA)
        entries = load_initial_data_txt(p)
        assert entries == [(0, 0, 0, 10), (0, 0, 1, 5)]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ConfigLoaderError, match="not found"):
            load_initial_data_txt(tmp_path / "missing.txt")

    def test_comments_ignored(self, tmp_path: Path) -> None:
        p = write(tmp_path, "comments.txt", "# header\n0,0,0,1\n# another\n")
        entries = load_initial_data_txt(p)
        assert entries == [(0, 0, 0, 1)]

    def test_blank_lines_ignored(self, tmp_path: Path) -> None:
        p = write(tmp_path, "blanks.txt", "0,0,0,1\n\n\n0,0,1,2\n")
        entries = load_initial_data_txt(p)
        assert entries == [(0, 0, 0, 1), (0, 0, 1, 2)]

    def test_wrong_field_count_raises(self, tmp_path: Path) -> None:
        p = write(tmp_path, "wrongcount.txt", "0,0,0\n")
        with pytest.raises(ConfigLoaderError, match="expected"):
            load_initial_data_txt(p)

    def test_non_integer_value_raises(self, tmp_path: Path) -> None:
        p = write(tmp_path, "nonint.txt", "0,0,0,abc\n")
        with pytest.raises(ConfigLoaderError, match="integers"):
            load_initial_data_txt(p)

    def test_negative_values_allowed(self, tmp_path: Path) -> None:
        p = write(tmp_path, "neg.txt", "0,0,0,-5\n")
        entries = load_initial_data_txt(p)
        assert entries == [(0, 0, 0, -5)]

    def test_empty_file_returns_empty_list(self, tmp_path: Path) -> None:
        p = write(tmp_path, "empty.txt", "")
        assert load_initial_data_txt(p) == []

    def test_whitespace_around_fields_stripped(self, tmp_path: Path) -> None:
        p = write(tmp_path, "spaces.txt", " 0 , 0 , 1 , 9 \n")
        entries = load_initial_data_txt(p)
        assert entries == [(0, 0, 1, 9)]

    def test_error_includes_line_number(self, tmp_path: Path) -> None:
        p = write(tmp_path, "lineno.txt", "0,0,0,1\nbad_line\n")
        with pytest.raises(ConfigLoaderError, match=":2:"):
            load_initial_data_txt(p)


# ---------------------------------------------------------------------------
# build_array_from_files — mesh size resolution
# ---------------------------------------------------------------------------

class TestMeshSizeResolution:
    def test_size_from_json(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps(VALID_INSTR))
        data = write(tmp_path, "d.txt", VALID_DATA)
        array = build_array_from_files(instr, data)
        assert array.rows == 2 and array.cols == 2

    def test_cli_overrides_json(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps(VALID_INSTR))
        data = write(tmp_path, "d.txt", VALID_DATA)
        array = build_array_from_files(instr, data, rows=5, cols=5)
        assert array.rows == 5 and array.cols == 5

    def test_inferred_from_program_and_data(self, tmp_path: Path) -> None:
        no_size = {"program": [{"1,2": {"opcode": "ADD", "src_a": 0,
                                          "src_b": 0, "dst": 0, "mux_sel": 0}}]}
        instr = write(tmp_path, "i.json", json.dumps(no_size))
        data = write(tmp_path, "d.txt", "0,0,0,1\n")
        array = build_array_from_files(instr, data)
        # highest index referenced is (1,2) -> mesh must be at least 2x3
        assert array.rows == 2 and array.cols == 3

    def test_inferred_from_data_only(self, tmp_path: Path) -> None:
        no_size = {"program": []}
        instr = write(tmp_path, "i.json", json.dumps(no_size))
        data = write(tmp_path, "d.txt", "3,4,0,1\n")
        array = build_array_from_files(instr, data)
        assert array.rows == 4 and array.cols == 5

    def test_minimal_1x1_inferred(self, tmp_path: Path) -> None:
        no_size = {"program": []}
        instr = write(tmp_path, "i.json", json.dumps(no_size))
        data = write(tmp_path, "d.txt", "0,0,0,1\n")
        array = build_array_from_files(instr, data)
        assert array.rows == 1 and array.cols == 1


# ---------------------------------------------------------------------------
# build_array_from_files — bounds validation
# ---------------------------------------------------------------------------

class TestBoundsValidation:
    def test_instruction_out_of_bounds_raises(self, tmp_path: Path) -> None:
        oob = {"rows": 1, "cols": 1,
               "program": [{"5,5": {"opcode": "ADD", "src_a": 0,
                                      "src_b": 0, "dst": 0, "mux_sel": 0}}]}
        instr = write(tmp_path, "i.json", json.dumps(oob))
        data = write(tmp_path, "d.txt", "0,0,0,1\n")
        with pytest.raises(ConfigLoaderError, match="outside"):
            build_array_from_files(instr, data)

    def test_initial_data_out_of_bounds_raises(self, tmp_path: Path) -> None:
        small = {"rows": 1, "cols": 1, "program": []}
        instr = write(tmp_path, "i.json", json.dumps(small))
        data = write(tmp_path, "d.txt", "9,9,0,1\n")
        with pytest.raises(ConfigLoaderError, match="outside"):
            build_array_from_files(instr, data)

    def test_register_index_out_of_rf_depth_raises(self, tmp_path: Path) -> None:
        small = {"rows": 1, "cols": 1, "program": []}
        instr = write(tmp_path, "i.json", json.dumps(small))
        data = write(tmp_path, "d.txt", "0,0,99,1\n")
        with pytest.raises(ConfigLoaderError, match="rf_depth"):
            build_array_from_files(instr, data, rf_depth=4)

    def test_cli_override_smaller_than_reference_raises(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps(VALID_INSTR))  # uses (0,0)
        big_data = write(tmp_path, "d.txt", "9,9,0,1\n")
        with pytest.raises(ConfigLoaderError, match="outside"):
            build_array_from_files(instr, big_data, rows=2, cols=2)


# ---------------------------------------------------------------------------
# build_array_from_files — end-to-end behavior
# ---------------------------------------------------------------------------

class TestEndToEndBuild:
    def test_full_pipeline_produces_correct_result(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps({
            "rows": 1, "cols": 1,
            "program": [{"0,0": {"opcode": "ADD", "src_a": 0, "src_b": 1,
                                   "dst": 2, "mux_sel": 0}}],
        }))
        data = write(tmp_path, "d.txt", "0,0,0,10\n0,0,1,5\n")
        array = build_array_from_files(instr, data)
        array.run(1)
        assert array.get_pe(0, 0).rf.read(2) == 15

    def test_torus_topology_applied(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps({
            "rows": 2, "cols": 2, "topology": "torus", "program": [],
        }))
        data = write(tmp_path, "d.txt", "")
        array = build_array_from_files(instr, data)
        assert array.get_topology() == "torus"

    def test_custom_rf_depth_applied(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps({
            "rows": 1, "cols": 1, "program": [],
        }))
        data = write(tmp_path, "d.txt", "")
        array = build_array_from_files(instr, data, rf_depth=8)
        assert array.get_pe(0, 0).rf.depth == 8

    def test_multi_pe_program_runs(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps(VALID_INSTR))
        data = write(tmp_path, "d.txt", VALID_DATA)
        array = build_array_from_files(instr, data)
        history = array.run(1)
        results = history[0]["results"]
        # PEArray.step() may key "results" by tuple[int,int] or by its
        # str() form depending on the implementation; accept either.
        value = results.get((0, 0), results.get(str((0, 0))))
        assert value == 15

    def test_no_initial_data_defaults_to_zero(self, tmp_path: Path) -> None:
        instr = write(tmp_path, "i.json", json.dumps({
            "rows": 1, "cols": 1,
            "program": [{"0,0": {"opcode": "ADD", "src_a": 0, "src_b": 1,
                                   "dst": 2, "mux_sel": 0}}],
        }))
        data = write(tmp_path, "d.txt", "")
        array = build_array_from_files(instr, data)
        array.run(1)
        assert array.get_pe(0, 0).rf.read(2) == 0