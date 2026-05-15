"""
Unit tests for the Network-on-Chip (NoC).

Verifies Mesh and Torus neighbor computation, output propagation,
topology locking (SYRS-MOD-001), and snapshot correctness.

Complies with: SYRS-FUN-006, SYRS-MOD-001, SYRS-REL-001
"""

import pytest

from archpilot_cgra.exceptions import SimulationException
from archpilot_cgra.noc import (
    TOPOLOGY_MESH,
    TOPOLOGY_TORUS,
    NoC,
)
from archpilot_cgra.processing_element import ProcessingElement


def make_grid(rows: int, cols: int) -> dict[tuple[int, int], ProcessingElement]:
    """Build a minimal PE grid for NoC propagation tests."""
    return {
        (r, c): ProcessingElement(pe_id=(r, c), rf_depth=4, data_width=32)
        for r in range(rows)
        for c in range(cols)
    }


class TestNoCInit:
    """Tests for NoC initialization."""

    def test_default_topology_is_mesh(self) -> None:
        noc = NoC(rows=2, cols=2)
        assert noc.topology == TOPOLOGY_MESH

    def test_torus_topology_stored(self) -> None:
        noc = NoC(rows=2, cols=2, topology=TOPOLOGY_TORUS)
        assert noc.topology == TOPOLOGY_TORUS

    def test_invalid_topology_raises(self) -> None:
        with pytest.raises(SimulationException):
            NoC(rows=2, cols=2, topology="ring")

    def test_invalid_rows_raises(self) -> None:
        with pytest.raises(SimulationException):
            NoC(rows=0, cols=2)

    def test_invalid_cols_raises(self) -> None:
        with pytest.raises(SimulationException):
            NoC(rows=2, cols=0)

    def test_initially_unlocked(self) -> None:
        noc = NoC(rows=2, cols=2)
        assert not noc.locked


class TestTopologyLock:
    """Tests for topology locking (SYRS-MOD-001)."""

    def test_lock_sets_locked_true(self) -> None:
        noc = NoC(rows=2, cols=2)
        noc.lock()
        assert noc.locked

    def test_unlock_sets_locked_false(self) -> None:
        noc = NoC(rows=2, cols=2)
        noc.lock()
        noc.unlock()
        assert not noc.locked

    def test_set_topology_when_unlocked(self) -> None:
        noc = NoC(rows=2, cols=2, topology=TOPOLOGY_MESH)
        noc.set_topology(TOPOLOGY_TORUS)
        assert noc.topology == TOPOLOGY_TORUS

    def test_set_topology_when_locked_raises(self) -> None:
        noc = NoC(rows=2, cols=2)
        noc.lock()
        with pytest.raises(SimulationException):
            noc.set_topology(TOPOLOGY_TORUS)

    def test_set_topology_invalid_raises(self) -> None:
        noc = NoC(rows=2, cols=2)
        with pytest.raises(SimulationException):
            noc.set_topology("star")


class TestMeshNeighbors:
    """Tests for Mesh neighbor computation."""

    def setup_method(self) -> None:
        self.noc = NoC(rows=3, cols=3, topology=TOPOLOGY_MESH)

    def test_center_has_all_four_neighbors(self) -> None:
        n = self.noc.get_neighbors(1, 1)
        assert n["norte"] == (0, 1)
        assert n["sur"] == (2, 1)
        assert n["este"] == (1, 2)
        assert n["oeste"] == (1, 0)

    def test_top_left_corner_has_no_north_or_west(self) -> None:
        n = self.noc.get_neighbors(0, 0)
        assert n["norte"] is None
        assert n["oeste"] is None
        assert n["sur"] == (1, 0)
        assert n["este"] == (0, 1)

    def test_bottom_right_corner_has_no_south_or_east(self) -> None:
        n = self.noc.get_neighbors(2, 2)
        assert n["sur"] is None
        assert n["este"] is None
        assert n["norte"] == (1, 2)
        assert n["oeste"] == (2, 1)

    def test_top_row_no_north(self) -> None:
        n = self.noc.get_neighbors(0, 1)
        assert n["norte"] is None

    def test_bottom_row_no_south(self) -> None:
        n = self.noc.get_neighbors(2, 1)
        assert n["sur"] is None

    def test_left_col_no_west(self) -> None:
        n = self.noc.get_neighbors(1, 0)
        assert n["oeste"] is None

    def test_right_col_no_east(self) -> None:
        n = self.noc.get_neighbors(1, 2)
        assert n["este"] is None


class TestTorusNeighbors:
    """Tests for Torus neighbor computation (wrap-around)."""

    def setup_method(self) -> None:
        self.noc = NoC(rows=3, cols=3, topology=TOPOLOGY_TORUS)

    def test_top_left_wraps_north(self) -> None:
        n = self.noc.get_neighbors(0, 0)
        assert n["norte"] == (2, 0)

    def test_top_left_wraps_west(self) -> None:
        n = self.noc.get_neighbors(0, 0)
        assert n["oeste"] == (0, 2)

    def test_bottom_right_wraps_south(self) -> None:
        n = self.noc.get_neighbors(2, 2)
        assert n["sur"] == (0, 2)

    def test_bottom_right_wraps_east(self) -> None:
        n = self.noc.get_neighbors(2, 2)
        assert n["este"] == (2, 0)

    def test_center_same_as_mesh(self) -> None:
        n = self.noc.get_neighbors(1, 1)
        assert n["norte"] == (0, 1)
        assert n["sur"] == (2, 1)
        assert n["este"] == (1, 2)
        assert n["oeste"] == (1, 0)

    def test_all_neighbors_are_not_none_in_torus(self) -> None:
        for r in range(3):
            for c in range(3):
                n = self.noc.get_neighbors(r, c)
                for direction, neighbor in n.items():
                    assert neighbor is not None, (
                        f"Torus neighbor {direction} at ({r},{c}) should not be None"
                    )


class TestMeshPropagation:
    """Tests for Mesh output propagation."""

    def test_center_pe_receives_all_neighbors(self) -> None:
        noc = NoC(rows=3, cols=3, topology=TOPOLOGY_MESH)
        grid = make_grid(3, 3)

        # Set known outputs on border PEs
        grid[(0, 1)].output_value = 10  # north of (1,1)
        grid[(2, 1)].output_value = 20  # south of (1,1)
        grid[(1, 2)].output_value = 30  # east  of (1,1)
        grid[(1, 0)].output_value = 40  # west  of (1,1)

        noc.propagate(grid)

        center = grid[(1, 1)]
        assert center.neighbor_inputs["norte"] == 10
        assert center.neighbor_inputs["sur"] == 20
        assert center.neighbor_inputs["este"] == 30
        assert center.neighbor_inputs["oeste"] == 40

    def test_corner_boundary_inputs_are_zero(self) -> None:
        noc = NoC(rows=3, cols=3, topology=TOPOLOGY_MESH)
        grid = make_grid(3, 3)
        noc.propagate(grid)
        corner = grid[(0, 0)]
        assert corner.neighbor_inputs["norte"] == 0
        assert corner.neighbor_inputs["oeste"] == 0

    def test_propagation_uses_snapshot(self) -> None:
        # Verify that PE A's output_value before propagate is used,
        # not any value set during the propagation phase itself.
        noc = NoC(rows=1, cols=2, topology=TOPOLOGY_MESH)
        grid = make_grid(1, 2)
        grid[(0, 0)].output_value = 99
        noc.propagate(grid)
        # (0,1) should receive (0,0)'s PRE-propagation output = 99
        assert grid[(0, 1)].neighbor_inputs["oeste"] == 99

    def test_one_dimensional_horizontal_propagation(self) -> None:
        noc = NoC(rows=1, cols=3, topology=TOPOLOGY_MESH)
        grid = make_grid(1, 3)
        grid[(0, 0)].output_value = 5
        grid[(0, 2)].output_value = 7
        noc.propagate(grid)
        assert grid[(0, 1)].neighbor_inputs["oeste"] == 5
        assert grid[(0, 1)].neighbor_inputs["este"] == 7


class TestTorusPropagation:
    """Tests for Torus output propagation (wrap-around)."""

    def test_left_edge_receives_right_edge_as_west(self) -> None:
        noc = NoC(rows=1, cols=3, topology=TOPOLOGY_TORUS)
        grid = make_grid(1, 3)
        grid[(0, 2)].output_value = 42
        noc.propagate(grid)
        # (0,0) west neighbor wraps to (0,2)
        assert grid[(0, 0)].neighbor_inputs["oeste"] == 42

    def test_top_edge_receives_bottom_edge_as_north(self) -> None:
        noc = NoC(rows=3, cols=1, topology=TOPOLOGY_TORUS)
        grid = make_grid(3, 1)
        grid[(2, 0)].output_value = 77
        noc.propagate(grid)
        # (0,0) north neighbor wraps to (2,0)
        assert grid[(0, 0)].neighbor_inputs["norte"] == 77

    def test_single_column_east_west_wrap(self) -> None:
        noc = NoC(rows=2, cols=2, topology=TOPOLOGY_TORUS)
        grid = make_grid(2, 2)
        grid[(0, 1)].output_value = 55
        noc.propagate(grid)
        # (0,0) east neighbor is (0,1); (0,1) west neighbor wraps to (0,0)
        assert grid[(0, 0)].neighbor_inputs["este"] == 55
