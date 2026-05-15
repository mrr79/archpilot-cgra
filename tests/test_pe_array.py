"""
Unit tests for PEArray.

Verifies mesh creation, cycle-accurate simulation, NoC integration,
ConfigMemory dispatch, topology API, activity tracing, and reset.

Complies with: SYRS-FUN-001, SYRS-FUN-005, SYRS-FUN-006,
               SYRS-FUN-007, SYRS-FUN-008, SYRS-MOD-001,
               SYRS-REL-001
"""

import pytest

from archpilot_cgra.config_memory import ConfigMemory
from archpilot_cgra.exceptions import SimulationException
from archpilot_cgra.noc import TOPOLOGY_MESH, TOPOLOGY_TORUS
from archpilot_cgra.pe_array import PEArray


ADD = {"opcode": "ADD", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}
MUL = {"opcode": "MUL", "src_a": 0, "src_b": 1, "dst": 2, "mux_sel": 0}
NOP = {"opcode": "NOP", "src_a": 0, "src_b": 0, "dst": 0, "mux_sel": 0}


class TestPEArrayInit:
    """Tests for PEArray initialization (SYRS-FUN-001)."""

    def test_create_4x4(self) -> None:
        array = PEArray(rows=4, cols=4)
        assert array.rows == 4
        assert array.cols == 4

    def test_all_pes_created(self) -> None:
        array = PEArray(rows=3, cols=3)
        assert len(array.get_all_pes()) == 9

    def test_initial_cycle_is_zero(self) -> None:
        array = PEArray(rows=2, cols=2)
        assert array.current_cycle == 0

    def test_invalid_rows_raises(self) -> None:
        with pytest.raises(SimulationException):
            PEArray(rows=0, cols=4)

    def test_invalid_cols_raises(self) -> None:
        with pytest.raises(SimulationException):
            PEArray(rows=4, cols=0)

    def test_default_topology_is_mesh(self) -> None:
        array = PEArray(rows=2, cols=2)
        assert array.get_topology() == TOPOLOGY_MESH

    def test_torus_topology_accepted(self) -> None:
        array = PEArray(rows=2, cols=2, topology=TOPOLOGY_TORUS)
        assert array.get_topology() == TOPOLOGY_TORUS

    def test_invalid_topology_raises(self) -> None:
        with pytest.raises(SimulationException):
            PEArray(rows=2, cols=2, topology="ring")

    def test_custom_config_memory_accepted(self) -> None:
        mem = ConfigMemory(rows=2, cols=2, config_sequence=[{(0, 0): ADD}])
        array = PEArray(rows=2, cols=2, config_memory=mem)
        assert array.config_memory is mem


class TestGetPE:
    """Tests for get_pe() and get_all_pes()."""

    def test_get_pe_correct_id(self) -> None:
        array = PEArray(rows=2, cols=2)
        assert array.get_pe(0, 1).pe_id == (0, 1)

    def test_get_pe_out_of_bounds_raises(self) -> None:
        array = PEArray(rows=2, cols=2)
        with pytest.raises(SimulationException):
            array.get_pe(5, 5)

    def test_get_all_pes_order(self) -> None:
        array = PEArray(rows=2, cols=2)
        pes = array.get_all_pes()
        ids = [pe.pe_id for pe in pes]
        assert ids == [(0, 0), (0, 1), (1, 0), (1, 1)]


class TestStep:
    """Tests for step() — single cycle execution (SYRS-FUN-005)."""

    def test_step_increments_cycle(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.step()
        assert array.current_cycle == 1

    def test_step_returns_cycle_key(self) -> None:
        array = PEArray(rows=2, cols=2)
        state = array.step()
        assert state["cycle"] == 0

    def test_step_returns_results_for_all_pes(self) -> None:
        array = PEArray(rows=2, cols=2)
        state = array.step()
        assert len(state["results"]) == 4

    def test_step_executes_instruction(self) -> None:
        seq = [{(0, 0): ADD}]
        mem = ConfigMemory(rows=2, cols=2, config_sequence=seq)
        array = PEArray(rows=2, cols=2, config_memory=mem)
        pe = array.get_pe(0, 0)
        pe.rf.write(0, 10)
        pe.rf.write(1, 5)
        array.step()
        assert pe.rf.read(2) == 15

    def test_step_locks_topology(self) -> None:
        array = PEArray(rows=2, cols=2)
        assert not array.noc.locked
        array.step()
        assert array.noc.locked

    def test_step_records_activity(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.step()
        assert len(array.activity_trace) == 1


class TestRun:
    """Tests for run() — multi-cycle execution (SYRS-FUN-005)."""

    def test_run_n_cycles(self) -> None:
        array = PEArray(rows=2, cols=2)
        history = array.run(5)
        assert len(history) == 5
        assert array.current_cycle == 5

    def test_run_zero_cycles_raises(self) -> None:
        array = PEArray(rows=2, cols=2)
        with pytest.raises(SimulationException):
            array.run(0)

    def test_run_accumulates_trace(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.run(3)
        assert len(array.activity_trace) == 3

    def test_run_100_cycles_4x4(self) -> None:
        # SYRS-PER-001 sanity check: 100 cycles must complete without error
        array = PEArray(rows=4, cols=4)
        history = array.run(100)
        assert len(history) == 100
        assert array.current_cycle == 100


class TestReset:
    """Tests for reset()."""

    def test_reset_clears_cycle(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.run(3)
        array.reset()
        assert array.current_cycle == 0

    def test_reset_clears_trace(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.run(3)
        array.reset()
        assert len(array.activity_trace) == 0

    def test_reset_unlocks_topology(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.step()
        assert array.noc.locked
        array.reset()
        assert not array.noc.locked

    def test_reset_clears_pe_registers(self) -> None:
        seq = [{(0, 0): ADD}]
        mem = ConfigMemory(rows=2, cols=2, config_sequence=seq)
        array = PEArray(rows=2, cols=2, config_memory=mem)
        pe = array.get_pe(0, 0)
        pe.rf.write(0, 99)
        array.step()
        array.reset()
        assert pe.rf.read(0) == 0

    def test_reset_allows_topology_change(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.step()
        array.reset()
        array.set_topology(TOPOLOGY_TORUS)
        assert array.get_topology() == TOPOLOGY_TORUS


class TestTopologyAPI:
    """Tests for topology API (SYRS-FUN-006, ACT-12)."""

    def test_set_topology_before_run(self) -> None:
        array = PEArray(rows=4, cols=4)
        array.set_topology(TOPOLOGY_TORUS)
        assert array.get_topology() == TOPOLOGY_TORUS

    def test_set_topology_during_run_raises(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.step()
        with pytest.raises(SimulationException):
            array.set_topology(TOPOLOGY_TORUS)

    def test_set_topology_after_reset_works(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.step()
        array.reset()
        array.set_topology(TOPOLOGY_TORUS)
        assert array.get_topology() == TOPOLOGY_TORUS

    def test_get_topology_returns_string(self) -> None:
        array = PEArray(rows=2, cols=2)
        assert isinstance(array.get_topology(), str)


class TestActivityTrace:
    """Tests for activity trace and summary (SYRS-FUN-007)."""

    def test_activity_summary_has_all_pes(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.run(2)
        summary = array.get_activity_summary()
        assert len(summary) == 4

    def test_active_pe_counted_in_summary(self) -> None:
        seq = [{(0, 0): ADD}, {(0, 0): ADD}]
        mem = ConfigMemory(rows=2, cols=2, config_sequence=seq)
        array = PEArray(rows=2, cols=2, config_memory=mem)
        pe = array.get_pe(0, 0)
        pe.rf.write(0, 1)
        pe.rf.write(1, 2)
        array.run(2)
        summary = array.get_activity_summary()
        assert summary[str((0, 0))] == 2

    def test_nop_pe_zero_in_summary(self) -> None:
        array = PEArray(rows=2, cols=2)
        array.run(3)
        summary = array.get_activity_summary()
        assert all(v == 0 for v in summary.values())


class TestNoCIntegration:
    """Tests for NoC data propagation within PEArray simulation."""

    def test_mesh_neighbor_propagation(self) -> None:
        # Cycle 0: PE(0,0) produces 7 (ADD 3+4)
        # Cycle 1: PE(1,0) reads norte neighbor (=7) + rf[1]=10 → 17
        seq_c0 = {
            (0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1,
                     "dst": 0, "mux_sel": 0},
            (1, 0): NOP,
        }
        seq_c1 = {
            (0, 0): NOP,
            (1, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1,
                     "dst": 2, "mux_sel": 1},  # mux_sel=1 → norte
        }
        mem = ConfigMemory(rows=2, cols=1, config_sequence=[seq_c0, seq_c1])
        array = PEArray(rows=2, cols=1, config_memory=mem)

        pe00 = array.get_pe(0, 0)
        pe10 = array.get_pe(1, 0)
        pe00.rf.write(0, 3)
        pe00.rf.write(1, 4)   # ADD: 3+4=7 at cycle 0
        pe10.rf.write(1, 10)  # operand_b

        array.step()   # cycle 0: pe00 produces 7
        array.step()   # cycle 1: pe10 reads norte=7, adds rf[1]=10 → 17

        assert pe10.rf.read(2) == 17

    def test_torus_wrap_propagation(self) -> None:
        # In a 2×1 torus, (0,0) north wraps to (1,0) and vice versa.
        # Cycle 0: PE(1,0) produces 5
        # Cycle 1: PE(0,0) reads norte (wraps to PE(1,0)) → 5
        seq_c0 = {
            (1, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1,
                     "dst": 0, "mux_sel": 0},
            (0, 0): NOP,
        }
        seq_c1 = {
            (1, 0): NOP,
            (0, 0): {"opcode": "ADD", "src_a": 0, "src_b": 1,
                     "dst": 2, "mux_sel": 1},  # norte wraps to (1,0)
        }
        mem = ConfigMemory(rows=2, cols=1, config_sequence=[seq_c0, seq_c1])
        array = PEArray(rows=2, cols=1, topology=TOPOLOGY_TORUS,
                        config_memory=mem)

        pe10 = array.get_pe(1, 0)
        pe00 = array.get_pe(0, 0)
        pe10.rf.write(0, 3)
        pe10.rf.write(1, 2)   # ADD: 3+2=5
        pe00.rf.write(1, 0)   # operand_b = 0 so result == norte input

        array.step()   # cycle 0: pe10 produces 5
        array.step()   # cycle 1: pe00 norte=5, 5+0=5

        assert pe00.rf.read(2) == 5


class TestGetArrayState:
    """Tests for get_array_state()."""

    def test_state_has_current_cycle(self) -> None:
        array = PEArray(rows=1, cols=1)
        state = array.get_array_state()
        assert state["current_cycle"] == 0

    def test_state_has_topology(self) -> None:
        array = PEArray(rows=1, cols=1)
        state = array.get_array_state()
        assert "topology" in state

    def test_state_has_all_pes(self) -> None:
        array = PEArray(rows=2, cols=2)
        state = array.get_array_state()
        assert len(state["pes"]) == 4
