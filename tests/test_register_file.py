"""
Unit tests for RegisterFile.

Verifies the configurable depth range (SYRS-FUN-004), read/write
operations, index validation, and reset behavior.

Complies with: SYRS-REL-001
"""

import pytest

from archpilot_cgra.exceptions import SimulationException
from archpilot_cgra.register_file import RegisterFile


class TestRegisterFile:
    """Test suite for the Register File."""

    # --- Creation ---

    def test_default_depth_is_four(self) -> None:
        rf = RegisterFile()
        assert len(rf) == 4

    def test_min_depth_two(self) -> None:
        rf = RegisterFile(depth=2)
        assert len(rf) == 2

    def test_max_depth_sixteen(self) -> None:
        rf = RegisterFile(depth=16)
        assert len(rf) == 16

    def test_depth_one_raises(self) -> None:
        with pytest.raises(SimulationException):
            RegisterFile(depth=1)

    def test_depth_seventeen_raises(self) -> None:
        with pytest.raises(SimulationException):
            RegisterFile(depth=17)

    def test_depth_zero_raises(self) -> None:
        with pytest.raises(SimulationException):
            RegisterFile(depth=0)

    # --- Initial state ---

    def test_initial_values_are_zero(self) -> None:
        rf = RegisterFile(depth=4)
        for i in range(4):
            assert rf.read(i) == 0

    # --- Read and write ---

    def test_write_and_read(self) -> None:
        rf = RegisterFile(depth=4)
        rf.write(2, 42)
        assert rf.read(2) == 42

    def test_write_negative_value(self) -> None:
        rf = RegisterFile(depth=4)
        rf.write(0, -7)
        assert rf.read(0) == -7

    def test_overwrite_value(self) -> None:
        rf = RegisterFile(depth=4)
        rf.write(1, 10)
        rf.write(1, 20)
        assert rf.read(1) == 20

    # --- Index validation ---

    def test_read_out_of_range_raises(self) -> None:
        rf = RegisterFile(depth=4)
        with pytest.raises(SimulationException):
            rf.read(4)

    def test_write_out_of_range_raises(self) -> None:
        # Both read and write go through _validate_index.
        # Both code paths must be tested symmetrically.
        rf = RegisterFile(depth=4)
        with pytest.raises(SimulationException):
            rf.write(4, 10)

    def test_write_negative_index_raises(self) -> None:
        rf = RegisterFile(depth=4)
        with pytest.raises(SimulationException):
            rf.write(-1, 10)

    def test_read_negative_index_raises(self) -> None:
        rf = RegisterFile(depth=4)
        with pytest.raises(SimulationException):
            rf.read(-1)

    # --- Reset ---

    def test_reset_clears_all_registers(self) -> None:
        rf = RegisterFile(depth=4)
        rf.write(0, 99)
        rf.write(3, 55)
        rf.reset()
        for i in range(4):
            assert rf.read(i) == 0

    # --- get_state ---

    def test_get_state_returns_all_values(self) -> None:
        rf = RegisterFile(depth=3)
        rf.write(1, 5)
        assert rf.get_state() == [0, 5, 0]

    def test_get_state_returns_copy(self) -> None:
        # Modifying the returned list must not affect the internal RF state.
        rf = RegisterFile(depth=4)
        rf.write(1, 7)
        state = rf.get_state()
        state[1] = 999
        assert rf.read(1) == 7
