"""
Unit tests for ConfigMemory.

Verifies instruction distribution, NOP fallback, thread safety,
sequence loading/appending, and mesh bounds validation.

Complies with: SYRS-FUN-008, SYRS-REL-001, SYRS-REL-002
"""

import threading

import pytest

from archpilot_cgra.config_memory import NOP_INSTRUCTION, ConfigMemory
from archpilot_cgra.exceptions import ConfigMemoryException, SimulationException


ADD = {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}
MUL = {"opcode": "MUL", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}
NOP = dict(NOP_INSTRUCTION)


class TestConfigMemoryInit:
    """Tests for ConfigMemory initialization."""

    def test_empty_memory_has_zero_cycles(self) -> None:
        mem = ConfigMemory(rows=2, cols=2)
        assert mem.num_cycles == 0

    def test_dimensions_stored(self) -> None:
        mem = ConfigMemory(rows=3, cols=4)
        assert mem.rows == 3
        assert mem.cols == 4

    def test_invalid_rows_raises(self) -> None:
        with pytest.raises(SimulationException):
            ConfigMemory(rows=0, cols=2)

    def test_invalid_cols_raises(self) -> None:
        with pytest.raises(SimulationException):
            ConfigMemory(rows=2, cols=0)

    def test_init_with_sequence(self) -> None:
        seq = [{(0, 0): ADD}]
        mem = ConfigMemory(rows=1, cols=1, config_sequence=seq)
        assert mem.num_cycles == 1

    def test_init_with_invalid_pe_id_raises(self) -> None:
        seq = [{(5, 5): ADD}]
        with pytest.raises(ConfigMemoryException):
            ConfigMemory(rows=2, cols=2, config_sequence=seq)


class TestGetInstruction:
    """Tests for get_instruction()."""

    def test_returns_nop_when_empty(self) -> None:
        mem = ConfigMemory(rows=1, cols=1)
        instr = mem.get_instruction(0, (0, 0))
        assert instr["opcode"] == "NOP"

    def test_returns_configured_instruction(self) -> None:
        mem = ConfigMemory(rows=1, cols=1, config_sequence=[{(0, 0): ADD}])
        instr = mem.get_instruction(0, (0, 0))
        assert instr["opcode"] == "ADD"

    def test_returns_nop_for_unconfigured_pe(self) -> None:
        mem = ConfigMemory(rows=2, cols=2, config_sequence=[{(0, 0): ADD}])
        instr = mem.get_instruction(0, (1, 1))
        assert instr["opcode"] == "NOP"

    def test_returns_nop_for_cycle_beyond_sequence(self) -> None:
        mem = ConfigMemory(rows=1, cols=1, config_sequence=[{(0, 0): ADD}])
        instr = mem.get_instruction(99, (0, 0))
        assert instr["opcode"] == "NOP"

    def test_returns_copy_not_reference(self) -> None:
        mem = ConfigMemory(rows=1, cols=1, config_sequence=[{(0, 0): ADD}])
        instr = mem.get_instruction(0, (0, 0))
        instr["opcode"] = "HACKED"
        assert mem.get_instruction(0, (0, 0))["opcode"] == "ADD"

    def test_multiple_cycles(self) -> None:
        seq = [{(0, 0): ADD}, {(0, 0): MUL}]
        mem = ConfigMemory(rows=1, cols=1, config_sequence=seq)
        assert mem.get_instruction(0, (0, 0))["opcode"] == "ADD"
        assert mem.get_instruction(1, (0, 0))["opcode"] == "MUL"


class TestDistribute:
    """Tests for distribute()."""

    def test_distribute_returns_all_pes(self) -> None:
        mem = ConfigMemory(rows=2, cols=3)
        result = mem.distribute(0)
        assert len(result) == 6

    def test_distribute_configured_pe_gets_instruction(self) -> None:
        seq = [{(0, 0): ADD, (1, 1): MUL}]
        mem = ConfigMemory(rows=2, cols=2, config_sequence=seq)
        dist = mem.distribute(0)
        assert dist[(0, 0)]["opcode"] == "ADD"
        assert dist[(1, 1)]["opcode"] == "MUL"

    def test_distribute_unconfigured_pe_gets_nop(self) -> None:
        seq = [{(0, 0): ADD}]
        mem = ConfigMemory(rows=2, cols=2, config_sequence=seq)
        dist = mem.distribute(0)
        assert dist[(0, 1)]["opcode"] == "NOP"
        assert dist[(1, 0)]["opcode"] == "NOP"

    def test_distribute_beyond_sequence_all_nop(self) -> None:
        mem = ConfigMemory(rows=2, cols=2)
        dist = mem.distribute(100)
        for instr in dist.values():
            assert instr["opcode"] == "NOP"

    def test_distribute_returns_copies(self) -> None:
        seq = [{(0, 0): ADD}]
        mem = ConfigMemory(rows=1, cols=1, config_sequence=seq)
        dist = mem.distribute(0)
        dist[(0, 0)]["opcode"] = "HACKED"
        assert mem.distribute(0)[(0, 0)]["opcode"] == "ADD"


class TestLoadSequence:
    """Tests for load_sequence()."""

    def test_load_sequence_replaces_existing(self) -> None:
        mem = ConfigMemory(rows=1, cols=1, config_sequence=[{(0, 0): ADD}])
        mem.load_sequence([{(0, 0): MUL}])
        assert mem.get_instruction(0, (0, 0))["opcode"] == "MUL"

    def test_load_sequence_updates_num_cycles(self) -> None:
        mem = ConfigMemory(rows=1, cols=1)
        mem.load_sequence([{(0, 0): ADD}, {(0, 0): MUL}])
        assert mem.num_cycles == 2

    def test_load_sequence_invalid_pe_raises(self) -> None:
        mem = ConfigMemory(rows=2, cols=2)
        with pytest.raises(ConfigMemoryException):
            mem.load_sequence([{(9, 9): ADD}])

    def test_load_sequence_does_not_corrupt_on_error(self) -> None:
        mem = ConfigMemory(rows=1, cols=1, config_sequence=[{(0, 0): ADD}])
        with pytest.raises(ConfigMemoryException):
            mem.load_sequence([{(0, 0): MUL}, {(5, 5): ADD}])
        # Original sequence must survive
        assert mem.get_instruction(0, (0, 0))["opcode"] == "ADD"


class TestAppendCycle:
    """Tests for append_cycle()."""

    def test_append_increases_num_cycles(self) -> None:
        mem = ConfigMemory(rows=1, cols=1)
        mem.append_cycle({(0, 0): ADD})
        assert mem.num_cycles == 1

    def test_append_multiple_cycles(self) -> None:
        mem = ConfigMemory(rows=1, cols=1)
        mem.append_cycle({(0, 0): ADD})
        mem.append_cycle({(0, 0): MUL})
        assert mem.num_cycles == 2
        assert mem.get_instruction(1, (0, 0))["opcode"] == "MUL"

    def test_append_invalid_pe_raises(self) -> None:
        mem = ConfigMemory(rows=1, cols=1)
        with pytest.raises(ConfigMemoryException):
            mem.append_cycle({(5, 5): ADD})


class TestClear:
    """Tests for clear()."""

    def test_clear_resets_num_cycles(self) -> None:
        mem = ConfigMemory(rows=1, cols=1, config_sequence=[{(0, 0): ADD}])
        mem.clear()
        assert mem.num_cycles == 0

    def test_clear_makes_all_nop(self) -> None:
        mem = ConfigMemory(rows=1, cols=1, config_sequence=[{(0, 0): ADD}])
        mem.clear()
        assert mem.get_instruction(0, (0, 0))["opcode"] == "NOP"


class TestThreadSafety:
    """Tests for thread-safe access (SYRS-REL-002)."""

    def test_concurrent_reads_are_safe(self) -> None:
        seq = [{(0, 0): ADD}]
        mem = ConfigMemory(rows=1, cols=1, config_sequence=seq)
        results: list[str] = []

        def reader() -> None:
            for _ in range(100):
                results.append(mem.get_instruction(0, (0, 0))["opcode"])

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert all(r == "ADD" for r in results)

    def test_concurrent_distribute_is_safe(self) -> None:
        mem = ConfigMemory(rows=2, cols=2)
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(50):
                    result = mem.distribute(0)
                    assert len(result) == 4
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
